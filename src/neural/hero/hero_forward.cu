// Hero device-resident forward for the `-hero` backend (see HERO_BACKEND.md).
// Weights upload ONCE at construction and are SHARED (read-only). Per-forward
// scratch lives in a POOL of Ctx (each with its own cublas handle + streams +
// buffers), so multiple evals from lc0's search threads OVERLAP on the GPU
// instead of serializing on one mutex — the engine-fill profile found the GPU
// ~65% idle during search; a Ctx pool keeps it fed. Everything runs on a
// per-Ctx stream (NOT the default stream) so the overlap is real.
// HERO_SLOTS (default 3) sets the pool size; =1 is the old serial path (A/B).
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "neural/hero/hero_forward.h"
#include "neural/hero/hero_weights.h"
#include "neural/backends/cuda/kernels.h"

using namespace lczero::cudnn_backend;
using half_t = half;

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"Hero CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s){fprintf(stderr,"Hero CUBLAS %d: %d\n",__LINE__,(int)s);exit(1);} }while(0)

namespace lczero {
namespace hero {

static void gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                 int m, int n, int k, float a, const half_t* A, int lda,
                 const half_t* B, int ldb, float b, half_t* C, int ldc) {
  CB(cublasGemmEx(h, ta, tb, m, n, k, &a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
                  &b, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static half_t* up_f(const std::vector<float>& v) {  // fp32 -> fp16 device (construction only)
  if (v.empty()) return nullptr;
  half_t* d; CK(cudaMalloc(&d, v.size() * sizeof(half_t)));
  float* tmp; CK(cudaMalloc(&tmp, v.size() * sizeof(float)));
  CK(cudaMemcpy(tmp, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted(d, tmp, (int)v.size(), 0);
  CK(cudaDeviceSynchronize()); CK(cudaFree(tmp));
  return d;
}

static std::vector<std::vector<float>> geo_basis() {
  std::vector<std::vector<float>> g(18, std::vector<float>(4096));
  auto R=[&](int s){return s/8;}; auto F=[&](int s){return s%8;};
  for (int i=0;i<64;i++) for (int j=0;j<64;j++) {
    int dR=R(i)-R(j), dF=F(i)-F(j), aR=abs(dR), aF=abs(dF);
    int dist=aR>aF?aR:aF;
    bool ray=((dR==0)||(dF==0)||(aR==aF))&&(dist>0);
    bool diag=(aR==aF)&&(dR!=0);
    int x=i*64+j; int k=0;
    auto set=[&](float v){ g[k++][x]=v; };
    set(((aR==2&&aF==1)||(aR==1&&aF==2))?1.f:0.f);
    set(dist==1?1.f:0.f);
    set(ray? 1.f/(dist+1) : 0.f);
    set(i==j?1.f:0.f);
    set(-(float)dist);
    set(1.f);
    set((dF==0&&dR<0)?1.f:0.f); set((dF==0&&dR>0)?1.f:0.f);
    set((dR==0&&dF<0)?1.f:0.f); set((dR==0&&dF>0)?1.f:0.f);
    set((diag&&dR<0&&dF<0)?1.f:0.f); set((diag&&dR<0&&dF>0)?1.f:0.f);
    set((diag&&dR>0&&dF<0)?1.f:0.f); set((diag&&dR>0&&dF>0)?1.f:0.f);
    set((dR==-1&&dF==0)?1.f:0.f); set((dR==-1&&aF==1)?1.f:0.f);
    set((dR==1&&dF==0)?1.f:0.f);  set((dR==1&&aF==1)?1.f:0.f);
  }
  return g;
}

__global__ void k_gather(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)r*d+c]=in[(size_t)idx[r]*d+c]; }
__global__ void k_scatter(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)idx[r]*d+c]=in[(size_t)r*d+c]; }
__global__ void k_route13(int* route,const half_t* planes,int N){
  int n=blockIdx.x, s=threadIdx.x;
  int best=-1; float bv=0.f; bool any=false;
  for(int c=0;c<12;c++){ float v=__half2float(planes[((size_t)n*112+c)*64+s]);
    if(v>0.f){any=true; if(best<0||v>bv){bv=v;best=c;}} }
  route[(size_t)n*64+s] = any? best+1 : 0;
}
__global__ void k_hist(int* cnt,const int* route,int R){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r<R) atomicAdd(&cnt[route[r]],1); }
__global__ void k_offsets(int* off,const int* cnt,int E){
  if(threadIdx.x) return; off[0]=0; for(int e=0;e<E;e++) off[e+1]=off[e]+cnt[e]; }
__global__ void k_order(int* order,int* cursor,const int* off,const int* route,int R){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R) return;
  int e=route[r]; order[off[e]+atomicAdd(&cursor[e],1)]=r; }
__global__ void k_promo(half_t* po,const half_t* kp,const half_t* PPO,int N,int pd){
  int n=blockIdx.x, f=blockIdx.y, c=threadIdx.x; if(c>=4) return;
  const half_t* kr=kp+((size_t)n*64+56+f)*pd; const half_t* pr=PPO+(size_t)c*pd;
  float s=0; for(int e=0;e<pd;e++) s+=__half2float(kr[e])*__half2float(pr[e]);
  po[((size_t)n*8+f)*4+c]=__float2half(s); }
__global__ void k_pol_gather(float* pol,const half_t* sc,const half_t* po,const int* g,int N){
  int n=blockIdx.x, m=blockIdx.y*blockDim.x+threadIdx.x; if(m>=1858) return;
  int idx=g[m]; float v;
  if(idx<4096) v=__half2float(sc[(size_t)n*4096+idx]);
  else { int p=idx-4096, rr=p/24, f=(p%24)/3, c=p%3;
    float base=__half2float(sc[(size_t)n*4096+(48+rr)*64+(56+f)]);
    v=base+__half2float(po[((size_t)n*8+f)*4+c])+__half2float(po[((size_t)n*8+f)*4+3]); }
  pol[(size_t)n*1858+m]=v; }
__global__ void k_wdl_mean(float* wdl,const half_t* vout,int N){
  int n=blockIdx.x, c=threadIdx.x; if(c>=3) return;
  float s=0; for(int i=0;i<64;i++) s+=__half2float(vout[((size_t)n*64+i)*3+c]);
  wdl[(size_t)n*3+c]=s/64.f; }

// ============================ the forward object ============================
struct HeroForward::Impl {
  HeroWeights w;
  int d,L,H,hd,dff,E,ed,pd; float alpha; int device=0;
  static const int NS = 8;    // streams for concurrent expert FFN inside a Ctx
  int nsUse = NS;
  // ---- shared weights (device, uploaded once, read-only) ----
  half_t *pp0,*pp1,*emb,*eln,*eu,*edn,*efln,*zbuf;
  struct LW { half_t *qw,*kw,*vw,*ow,*l1g,*l2g,*up,*dn,*bias; };
  std::vector<LW> lw;
  half_t *h_pe,*h_peb,*h_pq,*h_pqb,*h_pk,*h_pkb,*h_ve,*h_vq,*h_vk,*h_vv,*h_ppo;
  int* dGather=nullptr; std::once_flag gather_once;

  // ---- per-forward context (pooled) ----
  struct Ctx {
    Impl* I; cublasHandle_t cub; cudaStream_t main; cudaStream_t streams[NS];
    cudaEvent_t ev_gather, ev_done[NS];
    int capN=0;
    half_t *dPlanes,*dFlat,*pos128,*pos8192,*cat240,*emb_d,*e_out,*up_h,*dn_h,*x;
    half_t *qd,*kd,*vd,*po,*attn,*xa,*xs,*ffgh,*ys,*ffn;
    half_t *tp,*qp,*kp,*scp,*promo,*tv,*qv,*kv,*vvh,*scv,*vout;
    float *tmpP,*tmpF,*d_pol,*d_wdl;
    int *dOrder,*dRoute,*dCnt,*dCursor,*dOff;
    void setup(Impl* imp){ I=imp;
      CK(cudaSetDevice(I->device));
      CB(cublasCreate(&cub)); CB(cublasSetMathMode(cub,CUBLAS_TENSOR_OP_MATH));
      CK(cudaStreamCreate(&main)); CB(cublasSetStream(cub,main));
      for(int s=0;s<NS;s++){ CK(cudaStreamCreate(&streams[s])); cudaEventCreateWithFlags(&ev_done[s],cudaEventDisableTiming); }
      cudaEventCreateWithFlags(&ev_gather,cudaEventDisableTiming);
      CK(cudaMalloc(&dCnt,I->E*sizeof(int))); CK(cudaMalloc(&dCursor,I->E*sizeof(int))); CK(cudaMalloc(&dOff,(I->E+1)*sizeof(int)));
      ensure(512);
    }
    void ensure(int N){
      if(N<=capN) return; int d=I->d,ed=I->ed,pd=I->pd,dff=I->dff;
      if(capN){ for(half_t* p:{dPlanes,dFlat,pos128,pos8192,cat240,emb_d,e_out,up_h,dn_h,x,
                               qd,kd,vd,po,attn,xa,xs,ffgh,ys,ffn,tp,qp,kp,scp,promo,tv,qv,kv,vvh,scv,vout}) cudaFree(p);
                cudaFree(tmpP);cudaFree(tmpF);cudaFree(dOrder);cudaFree(dRoute);cudaFree(d_pol);cudaFree(d_wdl); }
      const size_t T=(size_t)N*64*d, R=(size_t)N*64;
      auto A=[&](half_t** p,size_t n){ CK(cudaMalloc(p,n*sizeof(half_t))); };
      A(&dPlanes,(size_t)N*112*64);A(&dFlat,(size_t)N*768);A(&pos128,(size_t)N*128);A(&pos8192,(size_t)N*8192);
      A(&cat240,R*240);A(&emb_d,T);A(&e_out,T);A(&up_h,R*ed);A(&dn_h,T);A(&x,T);A(&qd,T);A(&kd,T);A(&vd,T);A(&po,T);
      A(&attn,T);A(&xa,T);A(&xs,T);A(&ffgh,R*dff);A(&ys,T);A(&ffn,T);
      A(&tp,R*(size_t)pd);A(&qp,R*(size_t)pd);A(&kp,R*(size_t)pd);A(&scp,(size_t)N*4096);A(&promo,(size_t)N*8*4);
      A(&tv,R*(size_t)pd);A(&qv,R*(size_t)pd);A(&kv,R*(size_t)pd);A(&vvh,R*3);A(&scv,(size_t)N*4096);A(&vout,R*3);
      CK(cudaMalloc(&tmpP,(size_t)N*112*64*sizeof(float))); CK(cudaMalloc(&tmpF,(size_t)N*768*sizeof(float)));
      CK(cudaMalloc(&dOrder,R*sizeof(int)));CK(cudaMalloc(&dRoute,R*sizeof(int)));
      CK(cudaMalloc(&d_pol,(size_t)N*1858*sizeof(float)));CK(cudaMalloc(&d_wdl,(size_t)N*3*sizeof(float)));
      capN=N;
    }
    void head_gemm(half_t* W,int in,int out,half_t* bias,bool mish,half_t* xin,int rows,half_t* od){
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,1.f,W,in,xin,in,0.f,od,out);
      if(bias) addBiasBatched<half_t>(od,od,bias,1,rows,out,mish?ACTIVATION_MISH:ACTIVATION_NONE,main);
      else if(mish) addBiasBatched<half_t>(od,od,I->zbuf,1,rows,out,ACTIVATION_MISH,main);
    }
  };
  std::vector<Ctx> ctxs;
  std::mutex pool_mtx; std::condition_variable pool_cv; std::vector<int> freelist;
  int acquire(){ std::unique_lock<std::mutex> lk(pool_mtx); pool_cv.wait(lk,[&]{return !freelist.empty();});
    int c=freelist.back(); freelist.pop_back(); return c; }
  void release(int c){ { std::lock_guard<std::mutex> lk(pool_mtx); freelist.push_back(c);} pool_cv.notify_one(); }

  Impl(const HeroWeights& wt, int gpu) : w(wt), device(gpu) {
    CK(cudaSetDevice(device));
    d=w.d; L=w.layers; H=w.heads; hd=w.hd; dff=w.dff; E=w.classes; ed=w.embed_dff; pd=w.pol_d;
    alpha = powf(2.f*L, -0.25f);
    if (const char* e=getenv("HERO_FFN_STREAMS")){ nsUse=atoi(e); if(nsUse<1)nsUse=1; if(nsUse>NS)nsUse=NS; }
    pp0=up_f(w.preproc0_w); pp1=up_f(w.preproc1_w); emb=up_f(w.embed_w); eln=up_f(w.embed_ln_g);
    eu=up_f(w.embed_up_w); edn=up_f(w.embed_down_w); efln=up_f(w.embed_ffn_ln_g);
    std::vector<float> zeros((d>dff?d:dff),0.f); zbuf=up_f(zeros);
    auto geo = geo_basis();
    lw.resize(L);
    for (int li=0; li<L; li++) {
      auto& s=w.layer[li]; auto& t=lw[li];
      t.qw=up_f(s.q_w); t.kw=up_f(s.k_w); t.vw=up_f(s.v_w); t.ow=up_f(s.out_w);
      t.l1g=up_f(s.ln1_g); t.l2g=up_f(s.ln2_g); t.up=up_f(s.ffn_up); t.dn=up_f(s.ffn_down);
      std::vector<float> bias_hhh((size_t)H*64*64, 0.f);
      for (int h=0;h<H;h++) for (int i=0;i<64;i++) for (int j=0;j<64;j++){
        double b=s.free[(size_t)h*64*64+i*64+j];
        for (int k=0;k<18;k++) b += (double)s.alpha[h*18+k]*geo[k][i*64+j];
        bias_hhh[((size_t)h*64+i)*64+j]=(float)b;
      }
      t.bias=up_f(bias_hhh);
    }
    h_pe=up_f(w.pol_embed_w); h_peb=up_f(w.pol_embed_b);
    h_pq=up_f(w.pol_q_w); h_pqb=up_f(w.pol_q_b); h_pk=up_f(w.pol_k_w); h_pkb=up_f(w.pol_k_b);
    h_ve=up_f(w.val_embed_w); h_vq=up_f(w.val_q_w); h_vk=up_f(w.val_k_w); h_vv=up_f(w.val_v_w);
    h_ppo=up_f(w.pol_ppo_w);
    int slots=3; if(const char* e=getenv("HERO_SLOTS")){ slots=atoi(e); if(slots<1)slots=1; }
    ctxs.resize(slots);
    for(int i=0;i<slots;i++){ ctxs[i].setup(this); freelist.push_back(i); }
  }
};

// ---------------------------- public entry points ----------------------------
HeroForward::HeroForward(const HeroWeights& w, int gpu) : p_(new Impl(w, gpu)) {}
HeroForward::~HeroForward() { delete p_; }

void HeroForward::Run(const float* planes_nchw, const float* flat12, int N,
                      const std::vector<int>& gather, float* policy_out, float* wdl_out) {
  Impl& I=*p_; CK(cudaSetDevice(I.device));
  std::call_once(I.gather_once, [&]{ CK(cudaMalloc(&I.dGather,1858*sizeof(int)));
    CK(cudaMemcpy(I.dGather,gather.data(),1858*sizeof(int),cudaMemcpyHostToDevice)); });
  int ci=I.acquire(); Impl::Ctx& C=I.ctxs[ci];
  cublasHandle_t cub=C.cub; cudaStream_t S=C.main; C.ensure(N);
  const int d=I.d,H=I.H,hd=I.hd,dff=I.dff,E=I.E,ed=I.ed,pd=I.pd; const float al=I.alpha;
  const size_t T=(size_t)N*64*d; const int R=N*64;

  // upload planes (fp32 -> fp16) on this Ctx's stream (per-Ctx tmp buffers)
  CK(cudaMemcpyAsync(C.tmpP,planes_nchw,(size_t)N*112*64*sizeof(float),cudaMemcpyHostToDevice,S));
  copyTypeConverted(C.dPlanes,C.tmpP,(int)((size_t)N*112*64),S);
  CK(cudaMemcpyAsync(C.tmpF,flat12,(size_t)N*768*sizeof(float),cudaMemcpyHostToDevice,S));
  copyTypeConverted(C.dFlat,C.tmpF,(int)((size_t)N*768),S);

  // device routing -> off[] to host (need sizes for the per-expert gemms)
  std::vector<int> off(E+1);
  CK(cudaMemsetAsync(C.dCnt,0,E*sizeof(int),S)); CK(cudaMemsetAsync(C.dCursor,0,E*sizeof(int),S));
  k_route13<<<N,64,0,S>>>(C.dRoute,C.dPlanes,N);
  k_hist<<<(R+255)/256,256,0,S>>>(C.dCnt,C.dRoute,R);
  k_offsets<<<1,1,0,S>>>(C.dOff,C.dCnt,E);
  k_order<<<(R+255)/256,256,0,S>>>(C.dOrder,C.dCursor,C.dOff,C.dRoute,R);
  CK(cudaMemcpyAsync(off.data(),C.dOff,(E+1)*sizeof(int),cudaMemcpyDeviceToHost,S));
  CK(cudaStreamSynchronize(S));   // off[] ready; waits only THIS Ctx's stream

  // stem
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,128,N,768,1.f,I.pp0,768,C.dFlat,768,0.f,C.pos128,128);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,8192,N,128,1.f,I.pp1,128,C.pos128,128,0.f,C.pos8192,8192);
  inputPreprocessForAttentionBody<half_t>(C.cat240,C.dPlanes,C.pos8192,N,112,128,true,S);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,240,1.f,I.emb,240,C.cat240,240,0.f,C.emb_d,d);
  LayerNorm<half_t>(R,d,C.e_out,C.emb_d,I.zbuf,(half_t*)nullptr,I.eln,I.zbuf,1e-3f,1.f,ACTIVATION_MISH,S);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,ed,R,d,1.f,I.eu,d,C.e_out,d,0.f,C.up_h,ed);
  addBiasBatched<half_t>(C.up_h,C.up_h,I.zbuf,1,R,ed,ACTIVATION_MISH,S);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,ed,1.f,I.edn,ed,C.up_h,ed,0.f,C.dn_h,d);
  LayerNorm<half_t>(R,d,C.x,C.dn_h,I.zbuf,C.e_out,I.efln,I.zbuf,1e-3f,al,ACTIVATION_NONE,S);

  // trunk: 15 layers (fusedMHA + gather/scatter expert FFN)
  dim3 gd(R,(d+255)/256);
  for (int li=0; li<I.L; li++){
    auto& t=I.lw[li];
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.qw,d,C.x,d,0.f,C.qd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.kw,d,C.x,d,0.f,C.kd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.vw,d,C.x,d,0.f,C.vd,d);
    fusedMHA<half_t>(C.po,C.qd,C.kd,C.vd,t.bias,N,H,hd,S,true);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.ow,d,C.po,d,0.f,C.attn,d);
    LayerNorm<half_t>(R,d,C.xa,C.attn,I.zbuf,C.x,t.l1g,I.zbuf,1e-3f,al,ACTIVATION_NONE,S);
    // expert FFN: gather -> per-expert up/mish/down across NS streams -> scatter
    k_gather<<<gd,256,0,S>>>(C.xs,C.xa,C.dOrder,d);
    cudaEventRecord(C.ev_gather,S);
    for (int e=0;e<E;e++){ int m=off[e+1]-off[e]; if(!m) continue;
      cudaStream_t st=C.streams[e % I.nsUse];
      cudaStreamWaitEvent(st,C.ev_gather,0);
      cublasSetStream(cub,st);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,m,d,1.f,t.up+(size_t)e*dff*d,d,C.xs+(size_t)off[e]*d,d,0.f,C.ffgh+(size_t)off[e]*dff,dff);
      addBiasBatched<half_t>(C.ffgh+(size_t)off[e]*dff,C.ffgh+(size_t)off[e]*dff,I.zbuf,1,m,dff,ACTIVATION_MISH,st);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,m,dff,1.f,t.dn+(size_t)e*d*dff,dff,C.ffgh+(size_t)off[e]*dff,dff,0.f,C.ys+(size_t)off[e]*d,d); }
    for (int s=0;s<I.nsUse;s++){ cudaEventRecord(C.ev_done[s],C.streams[s]); cudaStreamWaitEvent(S,C.ev_done[s],0); }
    cublasSetStream(cub,S);
    k_scatter<<<gd,256,0,S>>>(C.ffn,C.ys,C.dOrder,d);
    LayerNorm<half_t>(R,d,C.x,C.ffn,I.zbuf,C.xa,t.l2g,I.zbuf,1e-3f,al,ACTIVATION_NONE,S);
  }

  // heads (fully device-resident, all on this Ctx's stream)
  const float sc=1.f/sqrtf((float)pd), z=0.f, o=1.f;
  C.head_gemm(I.h_pe,d,pd,I.h_peb,true,C.x,R,C.tp);
  C.head_gemm(I.h_pq,pd,pd,I.h_pqb,false,C.tp,R,C.qp);
  C.head_gemm(I.h_pk,pd,pd,I.h_pkb,false,C.tp,R,C.kp);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,pd,&sc,C.kp,CUDA_R_16F,pd,64*pd,C.qp,CUDA_R_16F,pd,64*pd,&z,C.scp,CUDA_R_16F,64,64*64,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  { dim3 pg(N,8); k_promo<<<pg,4,0,S>>>(C.promo,C.kp,I.h_ppo,N,pd); }
  { dim3 gg(N,(1858+255)/256); k_pol_gather<<<gg,256,0,S>>>(C.d_pol,C.scp,C.promo,I.dGather,N); }
  C.head_gemm(I.h_ve,d,pd,nullptr,true,C.x,R,C.tv);
  C.head_gemm(I.h_vq,pd,pd,nullptr,false,C.tv,R,C.qv);
  C.head_gemm(I.h_vk,pd,pd,nullptr,false,C.tv,R,C.kv);
  C.head_gemm(I.h_vv,pd,3,nullptr,false,C.tv,R,C.vvh);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,pd,&sc,C.kv,CUDA_R_16F,pd,64*pd,C.qv,CUDA_R_16F,pd,64*pd,&z,C.scv,CUDA_R_16F,64,64*64,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  Softmax<half_t>(N*64,64,C.scv,C.scv,(half_t*)nullptr,S);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,3,64,64,&o,C.vvh,CUDA_R_16F,3,64*3,C.scv,CUDA_R_16F,64,64*64,&z,C.vout,CUDA_R_16F,3,64*3,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  k_wdl_mean<<<N,3,0,S>>>(C.d_wdl,C.vout,N);
  CK(cudaMemcpyAsync(policy_out,C.d_pol,(size_t)N*1858*sizeof(float),cudaMemcpyDeviceToHost,S));
  CK(cudaMemcpyAsync(wdl_out,C.d_wdl,(size_t)N*3*sizeof(float),cudaMemcpyDeviceToHost,S));
  CK(cudaStreamSynchronize(S));   // results ready; only THIS Ctx's stream
  I.release(ci);
}

}  // namespace hero
}  // namespace lczero
