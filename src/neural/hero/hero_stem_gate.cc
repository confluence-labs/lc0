// Increment 3a gate: run Hero's STEM on the GPU from a .htw, compare to the
// torch oracle's post_stem.npy (1e-2). Self-contained — duplicates the two
// static helpers (cublasXgemm/allocAndUpload) and calls the public kernels.h
// kernels. Built + run on a GPU box (see hero-inference build script).
//
// Hero stem (hero.py, all bias-free, LN eps 1e-3 no-beta):
//   pos = preproc1( preproc0( planes[:,:,:12].reshape(N,768) ) ).reshape(N,64,128)   [no act]
//   e   = LN_mish( embed( cat(planes,pos) : N*64x240 ) )        (act=MISH, pre-norm)
//   x   = LN( e + alpha * embed_down( mish(embed_up(e)) ) )     (DeepNorm skip)
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "neural/hero/hero_weights.h"
#include "neural/backends/cuda/kernels.h"

using namespace lczero;
using namespace lczero::cudnn_backend;
using half_t = half;

// stub the one layers.cc symbol common_kernels needs (avoids compiling layers.cc)
namespace lczero { namespace cudnn_backend {
void CudaError(cudaError_t s, const char* f, const int& l) {
  if (s != cudaSuccess) { fprintf(stderr, "CUDA error %s (%s:%d)\n", cudaGetErrorString(s), f, l); exit(1); }
}
}}

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s){printf("CUBLAS %s @ %d: %d\n",#x,__LINE__,(int)s);exit(1);} } while(0)

// --- duplicated from layers.cc (static there) ---
static void gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                 int m, int n, int k, float alpha, const half_t* A, int lda,
                 const half_t* B, int ldb, float beta, half_t* C, int ldc) {
  __half al = __float2half(alpha), be = __float2half(beta);
  CB(cublasHgemm(h, ta, tb, m, n, k, &al, A, lda, B, ldb, &be, C, ldc));
}
static half_t* upload(const std::vector<float>& v, void* scratch) {
  if (v.empty()) return nullptr;
  half_t* d;
  CK(cudaMalloc(&d, v.size() * sizeof(half_t)));
  CK(cudaMemcpy(scratch, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted(d, (float*)scratch, (int)v.size(), 0);
  CK(cudaDeviceSynchronize());
  return d;
}

// --- minimal .npy reader (C-contiguous fp32/int32) ---
static std::vector<float> load_npy_f32(const std::string& p, std::vector<int>* shape=nullptr) {
  std::ifstream f(p, std::ios::binary);
  char magic[6]; f.read(magic, 6);
  uint8_t maj, min; f.read((char*)&maj,1); f.read((char*)&min,1);
  uint16_t hlen; f.read((char*)&hlen,2);
  std::string hdr(hlen, 0); f.read(&hdr[0], hlen);
  if (shape) {                               // parse "'shape': (a, b, ...)"
    size_t s = hdr.find("(") + 1, e = hdr.find(")");
    std::string sh = hdr.substr(s, e - s);
    size_t pos = 0; shape->clear();
    while (pos < sh.size()) {
      while (pos < sh.size() && (sh[pos]==' '||sh[pos]==',')) pos++;
      if (pos >= sh.size()) break;
      shape->push_back(atoi(sh.c_str()+pos));
      while (pos < sh.size() && sh[pos]!=',' ) pos++;
    }
  }
  std::vector<char> rest((std::istreambuf_iterator<char>(f)), {});
  bool is_i4 = hdr.find("<i4") != std::string::npos;
  size_t n = rest.size() / 4;
  std::vector<float> out(n);
  for (size_t i=0;i<n;i++) {
    if (is_i4) { int32_t v; memcpy(&v,&rest[i*4],4); out[i]=(float)v; }
    else       { float v; memcpy(&v,&rest[i*4],4); out[i]=v; }
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: hero_stem_gate <file.htw> <oracle_dir>\n"); return 2; }
  hero::HeroWeights w = hero::LoadHeroWeights(argv[1]);
  const int d = w.d, ed = w.embed_dff;
  std::string od = argv[2];
  std::vector<int> psh;
  auto planes = load_npy_f32(od + "/planes.npy", &psh);   // (N,64,112) 0/1
  auto ref    = load_npy_f32(od + "/post_stem.npy");      // (N,64,d)
  const int N = psh[0];
  printf("N=%d d=%d embed_dff=%d  planes=%zu ref=%zu\n", N, d, ed, planes.size(), ref.size());

  cublasHandle_t cub; CB(cublasCreate(&cub));
  void* scratch; CK(cudaMalloc(&scratch, 64ull*1024*1024*sizeof(float)));

  // weights
  half_t *pp0=upload(w.preproc0_w,scratch), *pp1=upload(w.preproc1_w,scratch);
  half_t *emb=upload(w.embed_w,scratch), *eln=upload(w.embed_ln_g,scratch);
  half_t *eu=upload(w.embed_up_w,scratch), *edn=upload(w.embed_down_w,scratch), *efln=upload(w.embed_ffn_ln_g,scratch);
  std::vector<float> zeros(d,0.f); half_t* zbuf=upload(zeros,scratch);

  // planes on device as (N,64,112) half, and its 12-plane flatten (N,768)
  std::vector<float> flat12(N*768);
  for (int n=0;n<N;n++) for (int s=0;s<64;s++) for (int c=0;c<12;c++)
    flat12[n*768 + s*12 + c] = planes[(size_t)n*64*112 + s*112 + c];
  half_t *dPlanes=upload(planes,scratch), *dFlat=upload(flat12,scratch);

  // buffers
  half_t *pos128, *pos8192, *cat240, *emb_d, *e_out, *up_h, *dn_h, *x_out;
  CK(cudaMalloc(&pos128,  (size_t)N*128*sizeof(half_t)));
  CK(cudaMalloc(&pos8192, (size_t)N*8192*sizeof(half_t)));
  CK(cudaMalloc(&cat240,  (size_t)N*64*240*sizeof(half_t)));
  CK(cudaMalloc(&emb_d,   (size_t)N*64*d*sizeof(half_t)));
  CK(cudaMalloc(&e_out,   (size_t)N*64*d*sizeof(half_t)));
  CK(cudaMalloc(&up_h,    (size_t)N*64*ed*sizeof(half_t)));
  CK(cudaMalloc(&dn_h,    (size_t)N*64*d*sizeof(half_t)));
  CK(cudaMalloc(&x_out,   (size_t)N*64*d*sizeof(half_t)));

  // preproc0: (N,768)@preproc0[128,768]^T -> (N,128)   (W row-major [out,in], OP_T)
  gemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, 128, N, 768, 1.f, pp0, 768, dFlat, 768, 0.f, pos128, 128);
  // preproc1: (N,128)@preproc1[8192,128]^T -> (N,8192)
  gemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, 8192, N, 128, 1.f, pp1, 128, pos128, 128, 0.f, pos8192, 8192);
  // concat planes(N,64,112) + pos(N,64,128) -> cat(N,64,240) via inputPreprocess
  inputPreprocessForAttentionBody<half_t>(cat240, dPlanes, pos8192, N, 112, 128, true, 0);
  // embed: (N*64,240)@embed[d,240]^T -> (N*64,d)
  gemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, d, N*64, 240, 1.f, emb, 240, cat240, 240, 0.f, emb_d, d);
  // LN(mish(embed)): act=MISH pre-norm, no skip, no bias/beta
  LayerNorm<half_t>(N*64, d, e_out, emb_d, zbuf, (half_t*)nullptr, eln, zbuf, 1e-3f, 1.f, ACTIVATION_MISH, 0);
  // embed_ffn: up -> mish -> down -> LN(e + alpha*down)
  gemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, ed, N*64, d, 1.f, eu, d, e_out, d, 0.f, up_h, ed);
  addBiasBatched<half_t>(up_h, up_h, zbuf, 1, N*64, ed, ACTIVATION_MISH, 0);  // zero-bias mish (zbuf has d>=ed zeros)
  gemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, d, N*64, ed, 1.f, edn, ed, up_h, ed, 0.f, dn_h, d);
  float alpha = powf(2.f * w.layers, -0.25f);
  LayerNorm<half_t>(N*64, d, x_out, dn_h, zbuf, e_out, efln, zbuf, 1e-3f, alpha, ACTIVATION_NONE, 0);
  CK(cudaDeviceSynchronize());

  // compare
  std::vector<half_t> hout(N*64*d); CK(cudaMemcpy(hout.data(), x_out, hout.size()*sizeof(half_t), cudaMemcpyDeviceToHost));
  double worst=0, sae=0;
  for (size_t i=0;i<hout.size();i++){ double dd=fabs((double)__half2float(hout[i]) - ref[i]); worst=fmax(worst,dd); sae+=dd; }
  printf("STEM gate: worst|d|=%.4f  mean|d|=%.5f  vs post_stem  (%s)\n",
         worst, sae/hout.size(), worst < 1e-2 ? "PASS" : "FAIL");
  return worst < 1e-2 ? 0 : 1;
}
