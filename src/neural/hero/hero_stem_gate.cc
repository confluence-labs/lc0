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
  // fp16 operands, fp32 accumulate (as lc0's cublasXgemm) — Hgemm's fp16
  // accumulate overflows to inf on 768-term dots.
  CB(cublasGemmEx(h, ta, tb, m, n, k, &alpha, A, CUDA_R_16F, lda,
                  B, CUDA_R_16F, ldb, &beta, C, CUDA_R_16F, ldc,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
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
static half_t* upload_h(const std::vector<half_t>& v, void*) {  // half vector, no conversion
  half_t* d; CK(cudaMalloc(&d, v.size() * sizeof(half_t)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(half_t), cudaMemcpyHostToDevice));
  return d;
}
// hero.py geo_basis(): 18 static (from,to) masks, each 64x64 (row-major i*64+j)
static std::vector<std::vector<float>> geo_basis() {
  std::vector<std::vector<float>> g(18, std::vector<float>(4096));
  auto R=[&](int s){return s/8;}; auto F=[&](int s){return s%8;};
  for (int i=0;i<64;i++) for (int j=0;j<64;j++) {
    int dR=R(i)-R(j), dF=F(i)-F(j), aR=abs(dR), aF=abs(dF);
    int dist=aR>aF?aR:aF;
    bool ray=((dR==0)||(dF==0)||(aR==aF))&&(dist>0);
    bool diag=(aR==aF)&&(dR!=0);
    int x=i*64+j; float* c=nullptr; int k=0;
    auto set=[&](float v){ g[k++][x]=v; };
    set(((aR==2&&aF==1)||(aR==1&&aF==2))?1.f:0.f);   // 0 knight
    set(dist==1?1.f:0.f);                            // 1 king-step
    set(ray? 1.f/(dist+1) : 0.f);                    // 2 ray/(dist+1)
    set(i==j?1.f:0.f);                               // 3 eye
    set(-(float)dist);                               // 4 -dist
    set(1.f);                                        // 5 ones
    set((dF==0&&dR<0)?1.f:0.f); set((dF==0&&dR>0)?1.f:0.f);   // 6,7 file rays
    set((dR==0&&dF<0)?1.f:0.f); set((dR==0&&dF>0)?1.f:0.f);   // 8,9 rank rays
    set((diag&&dR<0&&dF<0)?1.f:0.f); set((diag&&dR<0&&dF>0)?1.f:0.f);  // 10,11
    set((diag&&dR>0&&dF<0)?1.f:0.f); set((diag&&dR>0&&dF>0)?1.f:0.f);  // 12,13
    set((dR==-1&&dF==0)?1.f:0.f); set((dR==-1&&aF==1)?1.f:0.f);        // 14,15
    set((dR==1&&dF==0)?1.f:0.f);  set((dR==1&&aF==1)?1.f:0.f);         // 16,17
    (void)c;
  }
  return g;
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
  int w = hdr.find("u1")!=std::string::npos ? 1
        : (hdr.find("i8")!=std::string::npos||hdr.find("f8")!=std::string::npos) ? 8 : 4;
  bool is_i = hdr.find("<i")!=std::string::npos || hdr.find("u1")!=std::string::npos;
  bool is_f8 = hdr.find("f8")!=std::string::npos;
  size_t n = rest.size() / w;
  std::vector<float> out(n);
  for (size_t i=0;i<n;i++) {
    const char* p2=&rest[i*w];
    if (w==1)        out[i]=(float)(uint8_t)p2[0];
    else if (is_f8)  { double v; memcpy(&v,p2,8); out[i]=(float)v; }
    else if (w==8)   { int64_t v; memcpy(&v,p2,8); out[i]=(float)v; }
    else if (is_i)   { int32_t v; memcpy(&v,p2,4); out[i]=(float)v; }
    else             { float v; memcpy(&v,p2,4); out[i]=v; }
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

  // oracle planes are NHWC (N,64,112); the concat kernel wants NCHW (N,112,64).
  std::vector<float> planes_nchw((size_t)N*112*64);
  for (int n=0;n<N;n++) for (int hw=0;hw<64;hw++) for (int c=0;c<112;c++)
    planes_nchw[(size_t)n*112*64 + c*64 + hw] = planes[(size_t)n*64*112 + hw*112 + c];
  // preproc input: x[:,:,:12].reshape(N,768), square-major [sq0_ch0..11, sq1_...]
  std::vector<float> flat12((size_t)N*768);
  for (int n=0;n<N;n++) for (int s=0;s<64;s++) for (int c=0;c<12;c++)
    flat12[(size_t)n*768 + s*12 + c] = planes[(size_t)n*64*112 + s*112 + c];
  half_t *dPlanes=upload(planes_nchw,scratch), *dFlat=upload(flat12,scratch);

  auto dbg = [&](const char* nm, half_t* p, size_t cnt){
    std::vector<half_t> h(cnt); cudaMemcpy(h.data(), p, cnt*sizeof(half_t), cudaMemcpyDeviceToHost);
    double mx=0; int nan=0; for (auto v: h){ float f=__half2float(v); if(f!=f||f==INFINITY||f==-INFINITY)nan++; else mx=fmax(mx,fabs((double)f)); }
    printf("  [%s] max|finite|=%.3f nan/inf=%d first=%.4f\n", nm, mx, nan, __half2float(h[0]));
  };
  dbg("in:dFlat", dFlat, (size_t)N*768);
  dbg("in:preproc0_w", pp0, w.preproc0_w.size());

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

  dbg("x_out(stem)", x_out, (size_t)N*64*d);

  // ================= 3b: layer-0 attention block (DeepNorm) =================
  const int H = w.heads, hd = w.hd;   // 32, 32 ; H*hd == d
  // static per-head bias[H,64,64] = free0 + sum_k alpha0[h,k]*geo_basis[k]
  auto geo = geo_basis();                              // [18][64][64] float
  const auto& A0 = w.layer[0].alpha;   // [H,18]
  const auto& F0 = w.layer[0].free;    // [H,64,64]
  std::vector<float> bias_bcast((size_t)N*H*64*64);
  for (int h=0; h<H; h++) for (int i=0;i<64;i++) for (int j=0;j<64;j++) {
    double b = F0[(size_t)h*64*64 + i*64 + j];
    for (int k=0;k<18;k++) b += (double)A0[h*18+k] * geo[k][i*64+j];
    for (int n=0;n<N;n++) bias_bcast[((size_t)n*H+h)*64*64 + i*64 + j] = (float)b;
  }
  half_t* dBias = upload(bias_bcast, scratch);

  // q,k,v = x_stem @ W^T  (N*64, d)
  half_t *qw=upload(w.layer[0].q_w,scratch), *kw=upload(w.layer[0].k_w,scratch),
         *vw=upload(w.layer[0].v_w,scratch), *ow=upload(w.layer[0].out_w,scratch),
         *l1g=upload(w.layer[0].ln1_g,scratch);
  float zero=0.f, one=1.f;
  half_t *qd,*kd,*vd,*attn_out; size_t T=(size_t)N*64*d;
  CK(cudaMalloc(&qd,T*sizeof(half_t))); CK(cudaMalloc(&kd,T*sizeof(half_t)));
  CK(cudaMalloc(&vd,T*sizeof(half_t))); CK(cudaMalloc(&attn_out,T*sizeof(half_t)));
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,qw,d,x_out,d,0.f,qd,d);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,kw,d,x_out,d,0.f,kd,d);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,vw,d,x_out,d,0.f,vd,d);
  CK(cudaDeviceSynchronize());

  // transpose (N,64,H,hd) -> contiguous per-head (N*H,64,hd) on host (gate only)
  auto to_heads=[&](half_t* src, std::vector<half_t>& dst){
    std::vector<half_t> h(T); cudaMemcpy(h.data(),src,T*sizeof(half_t),cudaMemcpyDeviceToHost);
    dst.resize(T);
    for(int n=0;n<N;n++)for(int s=0;s<64;s++)for(int hh=0;hh<H;hh++)for(int e=0;e<hd;e++)
      dst[(((size_t)n*H+hh)*64+s)*hd+e] = h[((size_t)n*64+s)*d + hh*hd + e];
  };
  std::vector<half_t> qh,kh,vh; to_heads(qd,qh); to_heads(kd,kh); to_heads(vd,vh);
  half_t *qt=upload_h(qh,scratch),*kt=upload_h(kh,scratch),*vt=upload_h(vh,scratch);
  half_t *scores,*ctx; CK(cudaMalloc(&scores,(size_t)N*H*64*64*sizeof(half_t)));
  CK(cudaMalloc(&ctx,T*sizeof(half_t)));
  // scores = (q @ k^T)/sqrt(hd)  per (n,h): OP_T/OP_N, m=64,n=64,k=hd (lc0 convention)
  float fac=1.f/sqrtf((float)hd);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,hd,&fac,
      kt,CUDA_R_16F,hd,64*hd, qt,CUDA_R_16F,hd,64*hd, &zero,
      scores,CUDA_R_16F,64,64*64, N*H, CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  Softmax<half_t>(N*H*64, 64, scores, scores, dBias, 0);   // +bias, softmax over last dim
  // ctx = scores @ v  per (n,h): OP_N/OP_N, m=hd,n=64,k=64
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,hd,64,64,&one,
      vt,CUDA_R_16F,hd,64*hd, scores,CUDA_R_16F,64,64*64, &zero,
      ctx,CUDA_R_16F,hd,64*hd, N*H, CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  CK(cudaDeviceSynchronize());
  // transpose ctx (N*H,64,hd) -> (N,64,H*hd), then out gemm
  half_t* pout;
  { std::vector<half_t> c(T); cudaMemcpy(c.data(),ctx,T*sizeof(half_t),cudaMemcpyDeviceToHost);
    std::vector<half_t> o(T);
    for(int n=0;n<N;n++)for(int s=0;s<64;s++)for(int hh=0;hh<H;hh++)for(int e=0;e<hd;e++)
      o[((size_t)n*64+s)*d + hh*hd + e] = c[(((size_t)n*H+hh)*64+s)*hd+e];
    pout=upload_h(o,scratch); }
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,ow,d,pout,d,0.f,attn_out,d);
  // DeepNorm LN1: x_attn = LN(x_stem + alpha*attn_out)
  half_t* x_attn; CK(cudaMalloc(&x_attn,T*sizeof(half_t)));
  LayerNorm<half_t>(N*64, d, x_attn, attn_out, zbuf, x_out, l1g, zbuf, 1e-3f, alpha, ACTIVATION_NONE, 0);
  CK(cudaDeviceSynchronize());
  dbg("scores(softmax)", scores, (size_t)N*H*64*64);
  dbg("attn_out", attn_out, T);
  dbg("x_attn", x_attn, T);

  // compare x_attn vs post_attn0
  auto refA = load_npy_f32(od + "/post_attn0.npy");
  std::vector<half_t> hout(T); CK(cudaMemcpy(hout.data(), x_attn, T*sizeof(half_t), cudaMemcpyDeviceToHost));
  double worst=0, sae=0, refmax=0;
  for (size_t i=0;i<T;i++){ double dd=fabs((double)__half2float(hout[i]) - refA[i]); worst=fmax(worst,dd); sae+=dd; refmax=fmax(refmax,fabs(refA[i])); }
  double rel = worst / refmax; bool pass = rel < 8e-3 && sae/T < 1.5e-3;
  printf("ATTN gate: worst|d|=%.4f mean|d|=%.5f worst_rel=%.4f (of %.1f) (%s)\n",
         worst, sae/T, rel, refmax, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
