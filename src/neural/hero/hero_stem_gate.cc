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
#include <algorithm>

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
  std::vector<float> zeros((d>w.dff?d:w.dff),0.f); half_t* zbuf=upload(zeros,scratch);  // covers LN(d) + mish(dff)

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
  // ================= 3b/c/d: loop all layers, then heads =================
  const int H=w.heads, hd=w.hd, E=w.classes, dff=w.dff;
  auto geo = geo_basis();
  auto routef = load_npy_f32(od + "/route.npy");   // (N,64) expert per square
  float zero=0.f, one=1.f; (void)zero; (void)one;
  size_t T=(size_t)N*64*d;
  std::vector<float> bias_bcast((size_t)N*H*64*64);

  auto do_layer = [&](int li, half_t* x)->half_t* {
    // --- attention ---
    const std::vector<float>& AL=w.layer[li].alpha; const std::vector<float>& FL=w.layer[li].free;
    for (int h=0;h<H;h++) for (int i=0;i<64;i++) for (int j=0;j<64;j++){
      double b=FL[(size_t)h*64*64+i*64+j];
      for (int k=0;k<18;k++) b+=(double)AL[h*18+k]*geo[k][i*64+j];
      for (int n=0;n<N;n++) bias_bcast[((size_t)n*H+h)*64*64+i*64+j]=(float)b;
    }
    half_t* dBias=upload(bias_bcast,scratch);
    half_t *qw=upload(w.layer[li].q_w,scratch),*kw=upload(w.layer[li].k_w,scratch),
           *vw=upload(w.layer[li].v_w,scratch),*ow=upload(w.layer[li].out_w,scratch),
           *l1g=upload(w.layer[li].ln1_g,scratch),*l2g=upload(w.layer[li].ln2_g,scratch);
    half_t *qd,*kd,*vd,*attn_out; CK(cudaMalloc(&qd,T*sizeof(half_t)));CK(cudaMalloc(&kd,T*sizeof(half_t)));
    CK(cudaMalloc(&vd,T*sizeof(half_t)));CK(cudaMalloc(&attn_out,T*sizeof(half_t)));
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,qw,d,x,d,0.f,qd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,kw,d,x,d,0.f,kd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,vw,d,x,d,0.f,vd,d);
    CK(cudaDeviceSynchronize());
    auto to_heads=[&](half_t* src,std::vector<half_t>& dst){
      std::vector<half_t> h(T); cudaMemcpy(h.data(),src,T*sizeof(half_t),cudaMemcpyDeviceToHost); dst.resize(T);
      for(int n=0;n<N;n++)for(int s=0;s<64;s++)for(int hh=0;hh<H;hh++)for(int e=0;e<hd;e++)
        dst[(((size_t)n*H+hh)*64+s)*hd+e]=h[((size_t)n*64+s)*d+hh*hd+e]; };
    std::vector<half_t> qh,kh,vh; to_heads(qd,qh);to_heads(kd,kh);to_heads(vd,vh);
    half_t *qt=upload_h(qh,scratch),*kt=upload_h(kh,scratch),*vt=upload_h(vh,scratch);
    half_t *scores,*ctx; CK(cudaMalloc(&scores,(size_t)N*H*64*64*sizeof(half_t)));CK(cudaMalloc(&ctx,T*sizeof(half_t)));
    float fac=1.f/sqrtf((float)hd), z=0.f, o=1.f;
    CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,hd,&fac,kt,CUDA_R_16F,hd,64*hd,qt,CUDA_R_16F,hd,64*hd,&z,scores,CUDA_R_16F,64,64*64,N*H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    Softmax<half_t>(N*H*64,64,scores,scores,dBias,0);
    CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,hd,64,64,&o,vt,CUDA_R_16F,hd,64*hd,scores,CUDA_R_16F,64,64*64,&z,ctx,CUDA_R_16F,hd,64*hd,N*H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize());
    half_t* pout; { std::vector<half_t> c(T); cudaMemcpy(c.data(),ctx,T*sizeof(half_t),cudaMemcpyDeviceToHost);
      std::vector<half_t> ob(T); for(int n=0;n<N;n++)for(int s=0;s<64;s++)for(int hh=0;hh<H;hh++)for(int e=0;e<hd;e++)
        ob[((size_t)n*64+s)*d+hh*hd+e]=c[(((size_t)n*H+hh)*64+s)*hd+e]; pout=upload_h(ob,scratch); }
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,N*64,d,1.f,ow,d,pout,d,0.f,attn_out,d);
    half_t* x_attn; CK(cudaMalloc(&x_attn,T*sizeof(half_t)));
    LayerNorm<half_t>(N*64,d,x_attn,attn_out,zbuf,x,l1g,zbuf,1e-3f,alpha,ACTIVATION_NONE,0);
    CK(cudaDeviceSynchronize());
    // --- expert FFN ---
    std::vector<half_t> xa(T); cudaMemcpy(xa.data(),x_attn,T*sizeof(half_t),cudaMemcpyDeviceToHost);
    std::vector<half_t> yh(T,__float2half(0.f));
    half_t *gin,*gh,*gout; CK(cudaMalloc(&gin,T*sizeof(half_t)));CK(cudaMalloc(&gh,(size_t)N*64*dff*sizeof(half_t)));CK(cudaMalloc(&gout,T*sizeof(half_t)));
    for(int e=0;e<E;e++){
      std::vector<int> rows; for(int r=0;r<N*64;r++) if((int)routef[r]==e) rows.push_back(r);
      if(rows.empty()) continue; int M=rows.size();
      std::vector<half_t> gi((size_t)M*d); for(int m=0;m<M;m++) memcpy(&gi[(size_t)m*d],&xa[(size_t)rows[m]*d],d*sizeof(half_t));
      cudaMemcpy(gin,gi.data(),(size_t)M*d*sizeof(half_t),cudaMemcpyHostToDevice);
      half_t* up=upload(std::vector<float>(&w.layer[li].ffn_up[(size_t)e*dff*d],&w.layer[li].ffn_up[(size_t)(e+1)*dff*d]),scratch);
      half_t* dn=upload(std::vector<float>(&w.layer[li].ffn_down[(size_t)e*d*dff],&w.layer[li].ffn_down[(size_t)(e+1)*d*dff]),scratch);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,M,d,1.f,up,d,gin,d,0.f,gh,dff);
      addBiasBatched<half_t>(gh,gh,zbuf,1,M,dff,ACTIVATION_MISH,0);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,M,dff,1.f,dn,dff,gh,dff,0.f,gout,d);
      CK(cudaDeviceSynchronize());
      std::vector<half_t> go((size_t)M*d); cudaMemcpy(go.data(),gout,(size_t)M*d*sizeof(half_t),cudaMemcpyDeviceToHost);
      for(int m=0;m<M;m++) memcpy(&yh[(size_t)rows[m]*d],&go[(size_t)m*d],d*sizeof(half_t));
      cudaFree(up);cudaFree(dn);
    }
    half_t* ffn_out; CK(cudaMalloc(&ffn_out,T*sizeof(half_t)));
    cudaMemcpy(ffn_out,yh.data(),T*sizeof(half_t),cudaMemcpyHostToDevice);
    half_t* x_next; CK(cudaMalloc(&x_next,T*sizeof(half_t)));
    LayerNorm<half_t>(N*64,d,x_next,ffn_out,zbuf,x_attn,l2g,zbuf,1e-3f,alpha,ACTIVATION_NONE,0);
    CK(cudaDeviceSynchronize());
    cudaFree(qd);cudaFree(kd);cudaFree(vd);cudaFree(attn_out);cudaFree(scores);cudaFree(ctx);
    cudaFree(x_attn);cudaFree(ffn_out);cudaFree(gin);cudaFree(gh);cudaFree(gout);
    cudaFree(dBias);cudaFree(qw);cudaFree(kw);cudaFree(vw);cudaFree(ow);cudaFree(l1g);cudaFree(l2g);
    cudaFree(qt);cudaFree(kt);cudaFree(vt);cudaFree(pout);
    return x_next;
  };

  half_t* x = x_out;
  for (int li=0; li<w.layers; li++) x = do_layer(li, x);
  dbg("trunk", x, T);

  auto refT = load_npy_f32(od + "/post_trunk.npy");
  std::vector<half_t> ht(T); cudaMemcpy(ht.data(), x, T*sizeof(half_t), cudaMemcpyDeviceToHost);
  double wt=0,st=0,rmt=0;
  for (size_t i=0;i<T;i++){ double dd=fabs((double)__half2float(ht[i])-refT[i]); wt=fmax(wt,dd); st+=dd; rmt=fmax(rmt,fabs(refT[i])); }
  bool passT = wt/rmt < 6e-2 && st/T < 3e-3;   // fp16 accumulated over 15 layers
  printf("TRUNK gate: worst|d|=%.4f mean|d|=%.5f worst_rel=%.4f (of %.1f) (%s)\n",
         wt, st/T, wt/rmt, rmt, passT ? "PASS" : "FAIL");
  if (!passT) return 1;
  printf("TRUNK OK — heads (3d)\n");

  // ===================== 3d: policy + value heads =====================
  const int pd = w.pol_d;
  auto gemm_dl = [&](const std::vector<float>& W,int in,int out,const std::vector<float>* bias,bool mish,
                     half_t* xin,int rows)->std::vector<float>{
    half_t* wt_=upload(W,scratch); half_t* od_; CK(cudaMalloc(&od_,(size_t)rows*out*sizeof(half_t)));
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,1.f,wt_,in,xin,in,0.f,od_,out);
    if (bias){ half_t* b=upload(*bias,scratch); addBiasBatched<half_t>(od_,od_,b,1,rows,out,mish?ACTIVATION_MISH:ACTIVATION_NONE,0); cudaFree(b); }
    else if (mish){ addBiasBatched<half_t>(od_,od_,zbuf,1,rows,out,ACTIVATION_MISH,0); }
    CK(cudaDeviceSynchronize());
    std::vector<half_t> h((size_t)rows*out); cudaMemcpy(h.data(),od_,h.size()*sizeof(half_t),cudaMemcpyDeviceToHost);
    std::vector<float> f(h.size()); for(size_t i=0;i<h.size();i++) f[i]=__half2float(h[i]);
    cudaFree(wt_);cudaFree(od_); return f;
  };
  auto gather = load_npy_f32(od + "/gather.npy");   // 1858 int

  // ---- policy ----
  auto tp = gemm_dl(w.pol_embed_w,d,pd,&w.pol_embed_b,true,x,N*64);      // mish(embed+b)
  half_t* tpd=upload(tp,scratch);
  auto qp = gemm_dl(w.pol_q_w,pd,pd,&w.pol_q_b,false,tpd,N*64);
  auto kp = gemm_dl(w.pol_k_w,pd,pd,&w.pol_k_b,false,tpd,N*64);
  const auto& PPO = w.pol_ppo_w;   // [4, pd] ; hero.py applies ppo to K (not t)
  double sc = 1.0/sqrt((double)pd);
  std::vector<float> polout((size_t)N*1858);
  for (int n=0;n<N;n++){
    std::vector<float> attn(4096);
    for(int i=0;i<64;i++)for(int j=0;j<64;j++){ double s=0; for(int c=0;c<pd;c++) s+=(double)qp[((size_t)n*64+i)*pd+c]*kp[((size_t)n*64+j)*pd+c]; attn[i*64+j]=(float)(s*sc); }
    // promotion (192): ppo(k[56+f]) -> po[4]; off24[f*3+c] = po[c] + po[3]
    float off24[24]; for(int f=0;f<8;f++){ double po[4];
      for(int c=0;c<4;c++){ double s=0; for(int e=0;e<pd;e++) s+=(double)kp[((size_t)n*64+56+f)*pd+e]*PPO[c*pd+e]; po[c]=s; }
      for(int c=0;c<3;c++) off24[f*3+c]=(float)(po[c]+po[3]); }
    std::vector<float> prom(192);
    for(int rr=0;rr<8;rr++)for(int f=0;f<8;f++)for(int c=0;c<3;c++) prom[rr*24+f*3+c]=attn[(48+rr)*64+(56+f)]+off24[f*3+c];
    std::vector<float> cat(4288); memcpy(cat.data(),attn.data(),4096*sizeof(float)); memcpy(cat.data()+4096,prom.data(),192*sizeof(float));
    for(int m=0;m<1858;m++) polout[(size_t)n*1858+m]=cat[(int)gather[m]];
  }
  // ---- value ----
  auto tv = gemm_dl(w.val_embed_w,d,pd,nullptr,true,x,N*64);            // mish(embed)
  half_t* tvd=upload(tv,scratch);
  auto qv = gemm_dl(w.val_q_w,pd,pd,nullptr,false,tvd,N*64);
  auto kv = gemm_dl(w.val_k_w,pd,pd,nullptr,false,tvd,N*64);
  auto vv = gemm_dl(w.val_v_w,pd,3,nullptr,false,tvd,N*64);             // (N*64,3)
  std::vector<float> wdlout((size_t)N*3,0.f);
  for (int n=0;n<N;n++){
    double acc[3]={0,0,0};
    for(int i=0;i<64;i++){
      double s[64],mx=-1e30; for(int j=0;j<64;j++){ double a=0; for(int c=0;c<pd;c++) a+=(double)qv[((size_t)n*64+i)*pd+c]*kv[((size_t)n*64+j)*pd+c]; s[j]=a*sc; mx=fmax(mx,s[j]); }
      double z=0; for(int j=0;j<64;j++){ s[j]=exp(s[j]-mx); z+=s[j]; }
      for(int c=0;c<3;c++){ double o=0; for(int j=0;j<64;j++) o+=s[j]/z*vv[((size_t)n*64+j)*3+c]; acc[c]+=o; }
    }
    for(int c=0;c<3;c++) wdlout[(size_t)n*3+c]=(float)(acc[c]/64.0);
  }
  // compare
  auto refP=load_npy_f32(od+"/policy.npy"), refW=load_npy_f32(od+"/wdl.npy");
  double pw=0,ps=0,prm=0; for(size_t i=0;i<polout.size();i++){double dd=fabs(polout[i]-refP[i]);pw=fmax(pw,dd);ps+=dd;prm=fmax(prm,fabs(refP[i]));}
  double vw=0,rmw=0; for(size_t i=0;i<wdlout.size();i++){vw=fmax(vw,fabs(wdlout[i]-refW[i]));rmw=fmax(rmw,fabs(refW[i]));}
  // chess-meaningful: does the top move match per position (and top-3 overlap)?
  int top1=0, top3=0;
  for (int n=0;n<N;n++){
    auto am=[&](const std::vector<float>& v){ int b=0; for(int m=1;m<1858;m++) if(v[(size_t)n*1858+m]>v[(size_t)n*1858+b]) b=m; return b; };
    // top-3 of ref
    int r0=am(refP); std::vector<int> t3; for(int t=0;t<3;t++){int b=-1;for(int m=0;m<1858;m++){if(std::find(t3.begin(),t3.end(),m)!=t3.end())continue; if(b<0||refP[(size_t)n*1858+m]>refP[(size_t)n*1858+b])b=m;} t3.push_back(b);}
    int g0=am(polout);
    if (g0==r0) top1++;
    if (std::find(t3.begin(),t3.end(),g0)!=t3.end()) top3++;
  }
  double pol_rel = ps/polout.size()/prm;
  bool pol_ok = (top1==N) && pol_rel<3e-3, val_ok=vw/rmw<3e-2;
  printf("POLICY gate: top1-match=%d/%d top3=%d/%d mean_rel=%.5f worst|d|=%.3f (%s)\n", top1,N,top3,N,pol_rel,pw, pol_ok?"PASS":"FAIL");
  printf("VALUE  gate: worst|d|=%.4f wdl0=[%.3f %.3f %.3f] ref=[%.3f %.3f %.3f] (%s)\n",
         vw, wdlout[0],wdlout[1],wdlout[2], refW[0],refW[1],refW[2], val_ok?"PASS":"FAIL");
  printf("FULL FORWARD: %s\n", (pol_ok&&val_ok)?"PASS":"FAIL");
  return (pol_ok&&val_ok)?0:1;
}