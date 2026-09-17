// Hero device-resident forward for the `-hero` backend (see HERO_BACKEND.md).
// Merges the two validated references: hero_stem_gate.cc (real weights, stem,
// policy/value heads, oracle-gated) and hero_bench_cuda.cu (device trunk with
// fusedMHA + gather/scatter expert FFN). All weights upload ONCE at construction;
// Run() does one batched forward: stem -> 15 layers -> policy + WDL.
//
// Correct-first scope: the trunk (the ~99% of compute) is fully device-resident
// and uses the validated fusedMHA attention. Routing (13-class = piece type) and
// the tiny attention-style heads are computed host-side for now — they are ~1%
// of FLOPs and are the next thing to move on-device once nps is measured.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "neural/hero/hero_forward.h"
#include "neural/hero/hero_weights.h"
#include "neural/backends/cuda/kernels.h"

// NOTE: unlike the standalone gate/bench, this builds as part of full lc0, so the
// real CudaError (layers.cc) is linked — do NOT stub it here (would collide).
using namespace lczero::cudnn_backend;
using half_t = half;

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"Hero CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s){fprintf(stderr,"Hero CUBLAS %d: %d\n",__LINE__,(int)s);exit(1);} }while(0)

namespace lczero {
namespace hero {

// ---- fp16 gemm (fp32 accumulate, as lc0's cublasXgemm) ----
static void gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                 int m, int n, int k, float a, const half_t* A, int lda,
                 const half_t* B, int ldb, float b, half_t* C, int ldc) {
  CB(cublasGemmEx(h, ta, tb, m, n, k, &a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
                  &b, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static half_t* up_f(const std::vector<float>& v) {  // upload fp32 -> fp16 device
  if (v.empty()) return nullptr;
  half_t* d; CK(cudaMalloc(&d, v.size() * sizeof(half_t)));
  float* tmp; CK(cudaMalloc(&tmp, v.size() * sizeof(float)));
  CK(cudaMemcpy(tmp, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted(d, tmp, (int)v.size(), 0);
  CK(cudaDeviceSynchronize()); CK(cudaFree(tmp));
  return d;
}

// hero.py geo_basis(): 18 static (from,to) masks, each 64x64 row-major i*64+j
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

// device kernels for the FFN gather/scatter (rows pre-sorted by expert host-side)
__global__ void k_gather(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)r*d+c]=in[(size_t)idx[r]*d+c]; }
__global__ void k_scatter(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)idx[r]*d+c]=in[(size_t)r*d+c]; }

// ============================ the forward object ============================
struct HeroForward::Impl {
  HeroWeights w;
  cublasHandle_t cub;
  int d, L, H, hd, dff, E, ed, pd;
  float alpha;                       // DeepNorm (2L)^-0.25
  // stem weights
  half_t *pp0,*pp1,*emb,*eln,*eu,*edn,*efln,*zbuf;
  // per-layer weights + precomputed static bias (H,64,64)
  struct LW { half_t *qw,*kw,*vw,*ow,*l1g,*l2g,*up,*dn; std::vector<float> bias_hhh; };
  std::vector<LW> lw;
  // head weights kept on host (heads are ~1% compute, done host-side for now)

  Impl(const HeroWeights& wt) : w(wt) {
    d=w.d; L=w.layers; H=w.heads; hd=w.hd; dff=w.dff; E=w.classes; ed=w.embed_dff; pd=w.pol_d;
    alpha = powf(2.f*L, -0.25f);
    CB(cublasCreate(&cub)); CB(cublasSetMathMode(cub, CUBLAS_TENSOR_OP_MATH));
    pp0=up_f(w.preproc0_w); pp1=up_f(w.preproc1_w); emb=up_f(w.embed_w); eln=up_f(w.embed_ln_g);
    eu=up_f(w.embed_up_w); edn=up_f(w.embed_down_w); efln=up_f(w.embed_ffn_ln_g);
    std::vector<float> zeros((d>dff?d:dff),0.f); zbuf=up_f(zeros);
    auto geo = geo_basis();
    lw.resize(L);
    for (int li=0; li<L; li++) {
      auto& s=w.layer[li]; auto& t=lw[li];
      t.qw=up_f(s.q_w); t.kw=up_f(s.k_w); t.vw=up_f(s.v_w); t.ow=up_f(s.out_w);
      t.l1g=up_f(s.ln1_g); t.l2g=up_f(s.ln2_g); t.up=up_f(s.ffn_up); t.dn=up_f(s.ffn_down);
      // static per-head bias = free + alpha_mix . geo  (batch-independent)
      t.bias_hhh.assign((size_t)H*64*64, 0.f);
      for (int h=0;h<H;h++) for (int i=0;i<64;i++) for (int j=0;j<64;j++){
        double b=s.free[(size_t)h*64*64+i*64+j];
        for (int k=0;k<18;k++) b += (double)s.alpha[h*18+k]*geo[k][i*64+j];
        t.bias_hhh[((size_t)h*64+i)*64+j]=(float)b;
      }
    }
  }

  // 13-class route: piece type per square from planes[:, :12] (argmax, 0 if empty).
  // planes_nchw is [N,112,64]; returns route[N*64] in {0..12}. (Swappable: 28/45-class
  // partitions slot in here — see routing-direction memory.)
  void route13(const float* planes_nchw, int N, std::vector<int>& route) {
    route.assign((size_t)N*64, 0);
    for (int n=0;n<N;n++) for (int s=0;s<64;s++) {
      int best=-1; float bv=0.f; bool any=false;
      for (int c=0;c<12;c++){ float v=planes_nchw[((size_t)n*112+c)*64+s]; if(v>0.f){any=true; if(best<0||v>bv){bv=v;best=c;}} }
      route[(size_t)n*64+s] = any ? best+1 : 0;
    }
  }

  // policy + value heads (host, mirrors hero_stem_gate.cc exactly). trunk is (N*64,d) fp32.
  void heads(const std::vector<float>& trunk, int N, const std::vector<int>& gather,
             std::vector<float>& policy, std::vector<float>& wdl);

  ~Impl() { /* process-lifetime weights; leave to teardown */ }
};

// host head gemm helper (upload W, gemm, optional bias+mish, download fp32)
static std::vector<float> head_gemm(cublasHandle_t cub, const std::vector<float>& W, int in, int out,
                                    const std::vector<float>* bias, bool mish,
                                    const std::vector<float>& xin, int rows, half_t* zbuf) {
  half_t* wt=up_f(W); half_t* xi=up_f(xin); half_t* od; CK(cudaMalloc(&od,(size_t)rows*out*sizeof(half_t)));
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,1.f,wt,in,xi,in,0.f,od,out);
  if (bias){ half_t* b=up_f(*bias); addBiasBatched<half_t>(od,od,b,1,rows,out,mish?ACTIVATION_MISH:ACTIVATION_NONE,0); cudaFree(b); }
  else if (mish){ addBiasBatched<half_t>(od,od,zbuf,1,rows,out,ACTIVATION_MISH,0); }
  CK(cudaDeviceSynchronize());
  std::vector<half_t> h((size_t)rows*out); cudaMemcpy(h.data(),od,h.size()*sizeof(half_t),cudaMemcpyDeviceToHost);
  std::vector<float> f(h.size()); for(size_t i=0;i<h.size();i++) f[i]=__half2float(h[i]);
  cudaFree(wt); cudaFree(xi); cudaFree(od); return f;
}

void HeroForward::Impl::heads(const std::vector<float>& trunk, int N, const std::vector<int>& gather,
                              std::vector<float>& policy, std::vector<float>& wdl) {
  double sc = 1.0/sqrt((double)pd);
  // ---- policy ----
  auto tp = head_gemm(cub,w.pol_embed_w,d,pd,&w.pol_embed_b,true,trunk,N*64,zbuf);
  auto qp = head_gemm(cub,w.pol_q_w,pd,pd,&w.pol_q_b,false,tp,N*64,zbuf);
  auto kp = head_gemm(cub,w.pol_k_w,pd,pd,&w.pol_k_b,false,tp,N*64,zbuf);
  const auto& PPO=w.pol_ppo_w;
  policy.assign((size_t)N*1858,0.f);
  for (int n=0;n<N;n++){
    std::vector<float> attn(4096);
    for(int i=0;i<64;i++)for(int j=0;j<64;j++){ double s=0; for(int c=0;c<pd;c++) s+=(double)qp[((size_t)n*64+i)*pd+c]*kp[((size_t)n*64+j)*pd+c]; attn[i*64+j]=(float)(s*sc); }
    float off24[24]; for(int f=0;f<8;f++){ double po[4];
      for(int c=0;c<4;c++){ double s=0; for(int e=0;e<pd;e++) s+=(double)kp[((size_t)n*64+56+f)*pd+e]*PPO[c*pd+e]; po[c]=s; }
      for(int c=0;c<3;c++) off24[f*3+c]=(float)(po[c]+po[3]); }
    std::vector<float> cat(4288); memcpy(cat.data(),attn.data(),4096*sizeof(float));
    for(int rr=0;rr<8;rr++)for(int f=0;f<8;f++)for(int c=0;c<3;c++) cat[4096+rr*24+f*3+c]=attn[(48+rr)*64+(56+f)]+off24[f*3+c];
    for(int m=0;m<1858;m++) policy[(size_t)n*1858+m]=cat[gather[m]];
  }
  // ---- value ----
  auto tv = head_gemm(cub,w.val_embed_w,d,pd,nullptr,true,trunk,N*64,zbuf);
  auto qv = head_gemm(cub,w.val_q_w,pd,pd,nullptr,false,tv,N*64,zbuf);
  auto kv = head_gemm(cub,w.val_k_w,pd,pd,nullptr,false,tv,N*64,zbuf);
  auto vv = head_gemm(cub,w.val_v_w,pd,3,nullptr,false,tv,N*64,zbuf);
  wdl.assign((size_t)N*3,0.f);
  for (int n=0;n<N;n++){
    double acc[3]={0,0,0};
    for(int i=0;i<64;i++){
      double s[64],mx=-1e30; for(int j=0;j<64;j++){ double a=0; for(int c=0;c<pd;c++) a+=(double)qv[((size_t)n*64+i)*pd+c]*kv[((size_t)n*64+j)*pd+c]; s[j]=a*sc; mx=fmax(mx,s[j]); }
      double z=0; for(int j=0;j<64;j++){ s[j]=exp(s[j]-mx); z+=s[j]; }
      for(int c=0;c<3;c++){ double o=0; for(int j=0;j<64;j++) o+=s[j]/z*vv[((size_t)n*64+j)*3+c]; acc[c]+=o; }
    }
    for(int c=0;c<3;c++) wdl[(size_t)n*3+c]=(float)(acc[c]/64.0);
  }
}

// ---------------------------- public entry points ----------------------------
HeroForward::HeroForward(const HeroWeights& w) : p_(new Impl(w)) {}
HeroForward::~HeroForward() { delete p_; }

void HeroForward::Run(const float* planes_nchw, const float* flat12, int N,
                      const std::vector<int>& gather, float* policy_out, float* wdl_out) {
  Impl& I=*p_; cublasHandle_t cub=I.cub;
  const int d=I.d,H=I.H,hd=I.hd,dff=I.dff,E=I.E,ed=I.ed; const float al=I.alpha;
  const size_t T=(size_t)N*64*d; const int R=N*64;

  half_t *dPlanes=up_f(std::vector<float>(planes_nchw,planes_nchw+(size_t)N*112*64));
  half_t *dFlat  =up_f(std::vector<float>(flat12,flat12+(size_t)N*768));

  // ---- stem: preproc -> concat -> embed(LN mish) -> embed_ffn(DeepNorm) ----
  half_t *pos128,*pos8192,*cat240,*emb_d,*e_out,*up_h,*dn_h,*x;
  CK(cudaMalloc(&pos128,(size_t)N*128*sizeof(half_t))); CK(cudaMalloc(&pos8192,(size_t)N*8192*sizeof(half_t)));
  CK(cudaMalloc(&cat240,(size_t)R*240*sizeof(half_t))); CK(cudaMalloc(&emb_d,T*sizeof(half_t)));
  CK(cudaMalloc(&e_out,T*sizeof(half_t))); CK(cudaMalloc(&up_h,(size_t)R*ed*sizeof(half_t)));
  CK(cudaMalloc(&dn_h,T*sizeof(half_t))); CK(cudaMalloc(&x,T*sizeof(half_t)));
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,128,N,768,1.f,I.pp0,768,dFlat,768,0.f,pos128,128);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,8192,N,128,1.f,I.pp1,128,pos128,128,0.f,pos8192,8192);
  inputPreprocessForAttentionBody<half_t>(cat240,dPlanes,pos8192,N,112,128,true,0);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,240,1.f,I.emb,240,cat240,240,0.f,emb_d,d);
  LayerNorm<half_t>(R,d,e_out,emb_d,I.zbuf,(half_t*)nullptr,I.eln,I.zbuf,1e-3f,1.f,ACTIVATION_MISH,0);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,ed,R,d,1.f,I.eu,d,e_out,d,0.f,up_h,ed);
  addBiasBatched<half_t>(up_h,up_h,I.zbuf,1,R,ed,ACTIVATION_MISH,0);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,ed,1.f,I.edn,ed,up_h,ed,0.f,dn_h,d);
  LayerNorm<half_t>(R,d,x,dn_h,I.zbuf,e_out,I.efln,I.zbuf,1e-3f,al,ACTIVATION_NONE,0);

  // ---- route (13-class piece) + host sort -> order/offsets per expert ----
  std::vector<int> route; I.route13(planes_nchw,N,route);
  std::vector<int> order(R), off(E+1,0), cnt(E,0);
  for (int r=0;r<R;r++) cnt[route[r]]++;
  for (int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
  { std::vector<int> cur(off.begin(),off.end()-1); for(int r=0;r<R;r++) order[cur[route[r]]++]=r; }
  int* dOrder; CK(cudaMalloc(&dOrder,R*sizeof(int))); CK(cudaMemcpy(dOrder,order.data(),R*sizeof(int),cudaMemcpyHostToDevice));

  // ---- trunk: 15 layers (fusedMHA attention + gather/scatter expert FFN) ----
  half_t *qd,*kd,*vd,*po,*attn,*xa,*xs,*ffgh,*ys,*ffn;
  CK(cudaMalloc(&qd,T*sizeof(half_t)));CK(cudaMalloc(&kd,T*sizeof(half_t)));CK(cudaMalloc(&vd,T*sizeof(half_t)));
  CK(cudaMalloc(&po,T*sizeof(half_t)));CK(cudaMalloc(&attn,T*sizeof(half_t)));CK(cudaMalloc(&xa,T*sizeof(half_t)));
  CK(cudaMalloc(&xs,T*sizeof(half_t)));CK(cudaMalloc(&ffgh,(size_t)R*dff*sizeof(half_t)));
  CK(cudaMalloc(&ys,T*sizeof(half_t)));CK(cudaMalloc(&ffn,T*sizeof(half_t)));
  half_t* dBias; CK(cudaMalloc(&dBias,(size_t)N*H*64*64*sizeof(half_t)));
  dim3 gd(R,(d+255)/256);
  std::vector<float> bias_bcast((size_t)N*H*64*64);
  for (int li=0; li<I.L; li++){
    auto& t=I.lw[li];
    // broadcast the static (H,64,64) bias to (N,H,64,64) for fusedMHA
    for (int n=0;n<N;n++) memcpy(&bias_bcast[(size_t)n*H*64*64], t.bias_hhh.data(), (size_t)H*64*64*sizeof(float));
    half_t* bcopy=up_f(bias_bcast); CK(cudaMemcpy(dBias,bcopy,(size_t)N*H*64*64*sizeof(half_t),cudaMemcpyDeviceToDevice)); cudaFree(bcopy);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.qw,d,x,d,0.f,qd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.kw,d,x,d,0.f,kd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.vw,d,x,d,0.f,vd,d);
    fusedMHA<half_t>(po, qd, kd, vd, dBias, N, H, hd, 0);   // validated fast attention
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.ow,d,po,d,0.f,attn,d);
    LayerNorm<half_t>(R,d,xa,attn,I.zbuf,x,t.l1g,I.zbuf,1e-3f,al,ACTIVATION_NONE,0);
    // expert FFN: gather by expert -> per-expert up/mish/down -> scatter
    k_gather<<<gd,256>>>(xs,xa,dOrder,d);
    for (int e=0;e<E;e++){ int m=off[e+1]-off[e]; if(!m) continue;
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,m,d,1.f,t.up+(size_t)e*dff*d,d,xs+(size_t)off[e]*d,d,0.f,ffgh+(size_t)off[e]*dff,dff);
      addBiasBatched<half_t>(ffgh+(size_t)off[e]*dff,ffgh+(size_t)off[e]*dff,I.zbuf,1,m,dff,ACTIVATION_MISH,0);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,m,dff,1.f,t.dn+(size_t)e*d*dff,dff,ffgh+(size_t)off[e]*dff,dff,0.f,ys+(size_t)off[e]*d,d); }
    k_scatter<<<gd,256>>>(ffn,ys,dOrder,d);
    LayerNorm<half_t>(R,d,x,ffn,I.zbuf,xa,t.l2g,I.zbuf,1e-3f,al,ACTIVATION_NONE,0);  // x reused as next input
  }
  CK(cudaDeviceSynchronize());

  // ---- download trunk (fp32) and run heads host-side ----
  std::vector<half_t> ht(T); cudaMemcpy(ht.data(),x,T*sizeof(half_t),cudaMemcpyDeviceToHost);
  std::vector<float> trunk(T); for(size_t i=0;i<T;i++) trunk[i]=__half2float(ht[i]);
  std::vector<float> policy, wdl; I.heads(trunk,N,gather,policy,wdl);
  memcpy(policy_out, policy.data(), policy.size()*sizeof(float));
  memcpy(wdl_out, wdl.data(), wdl.size()*sizeof(float));

  cudaFree(dPlanes);cudaFree(dFlat);cudaFree(pos128);cudaFree(pos8192);cudaFree(cat240);
  cudaFree(emb_d);cudaFree(e_out);cudaFree(up_h);cudaFree(dn_h);cudaFree(x);
  cudaFree(qd);cudaFree(kd);cudaFree(vd);cudaFree(po);cudaFree(attn);cudaFree(xa);cudaFree(xs);
  cudaFree(ffgh);cudaFree(ys);cudaFree(ffn);cudaFree(dBias);cudaFree(dOrder);
}

}  // namespace hero
}  // namespace lczero
