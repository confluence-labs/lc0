// Hero device-resident forward for the `-hero` backend (see HERO_BACKEND.md).
// Merges the two validated references: hero_stem_gate.cc (real weights, stem,
// policy/value heads, oracle-gated) and hero_bench_cuda.cu (device trunk with
// fusedMHA + gather/scatter expert FFN). ALL weights + scratch buffers upload
// ONCE at construction; Run() does one batched forward with NO host allocs and
// NO per-call weight uploads (the correct-first version re-did that every call
// and ran at ~46 nps — this is the perf pass).
//
// The trunk (~99% of compute) is fully device-resident with the validated
// fusedMHA attention. Routing (13-class = piece type) is host-side per batch
// (tiny) and the small attention-style heads run host-side on downloaded q/k
// (the validated math) — moved to device only if measured to bottleneck.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
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

static void gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                 int m, int n, int k, float a, const half_t* A, int lda,
                 const half_t* B, int ldb, float b, half_t* C, int ldc) {
  CB(cublasGemmEx(h, ta, tb, m, n, k, &a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
                  &b, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static half_t* up_f(const std::vector<float>& v) {  // upload fp32 -> fp16 device (construction only)
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

// device kernels: FFN gather/scatter (rows pre-sorted by expert) + bias broadcast
__global__ void k_gather(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)r*d+c]=in[(size_t)idx[r]*d+c]; }
__global__ void k_scatter(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)idx[r]*d+c]=in[(size_t)r*d+c]; }
// broadcast per-head bias (H,64,64) -> (N,H,64,64) for fusedMHA (batch-independent)
__global__ void k_bcast_bias(half_t* o,const half_t* b,int HB){  // HB = H*64*64
  int n=blockIdx.y, x=blockIdx.x*blockDim.x+threadIdx.x; if(x<HB) o[(size_t)n*HB+x]=b[x]; }
// promotion logits: po[n,f,c] = sum_e kp[n,56+f,e]*PPO[c,e]  (f 0..7, c 0..3)
__global__ void k_promo(half_t* po,const half_t* kp,const half_t* PPO,int N,int pd){
  int n=blockIdx.x, f=blockIdx.y, c=threadIdx.x; if(c>=4) return;
  const half_t* kr=kp+((size_t)n*64+56+f)*pd; const half_t* pr=PPO+(size_t)c*pd;
  float s=0; for(int e=0;e<pd;e++) s+=__half2float(kr[e])*__half2float(pr[e]);
  po[((size_t)n*8+f)*4+c]=__float2half(s); }
// policy gather: cat4288[i*64+j | 4096+promo] -> 1858 via the fixed lc0 map
__global__ void k_pol_gather(float* pol,const half_t* sc,const half_t* po,const int* g,int N){
  int n=blockIdx.x, m=blockIdx.y*blockDim.x+threadIdx.x; if(m>=1858) return;
  int idx=g[m]; float v;
  if(idx<4096) v=__half2float(sc[(size_t)n*4096+idx]);
  else { int p=idx-4096, rr=p/24, f=(p%24)/3, c=p%3;
    float base=__half2float(sc[(size_t)n*4096+(48+rr)*64+(56+f)]);
    v=base+__half2float(po[((size_t)n*8+f)*4+c])+__half2float(po[((size_t)n*8+f)*4+3]); }
  pol[(size_t)n*1858+m]=v; }
// WDL = mean over the 64 query rows of the value attention output (N,64,3)
__global__ void k_wdl_mean(float* wdl,const half_t* vout,int N){
  int n=blockIdx.x, c=threadIdx.x; if(c>=3) return;
  float s=0; for(int i=0;i<64;i++) s+=__half2float(vout[((size_t)n*64+i)*3+c]);
  wdl[(size_t)n*3+c]=s/64.f; }

// ---- device 28-class routing (mirrors route28() host logic, bit-exact) ----
// occ + piece(argmax) per square, from dPlanes [N,112,64] fp16 (NCHW)
__global__ void k_occ_piece(const half_t* pl, uint8_t* occ, uint8_t* piece, int N){
  int n=blockIdx.x, s=blockIdx.y*blockDim.x+threadIdx.x; if(s>=64) return;
  int pc=0; float bv=0; bool any=false;
  for(int c=0;c<12;c++){ float v=__half2float(pl[((size_t)n*112+c)*64+s]); if(v>0.f){any=true; if(pc==0||v>bv){bv=v;pc=c+1;}} }
  size_t i=(size_t)n*64+s; occ[i]=any?1:0; piece[i]=(uint8_t)pc; }
// attackers: one thread per (n,from), scatter attacks PER SET PLANE (matches the
// reference on random planes where a square may set >1 channel). att_* pre-zeroed.
__global__ void k_attackers(const half_t* pl,const uint8_t* occ,uint8_t* att_o,uint8_t* att_t,int N){
  int n=blockIdx.x, from=blockIdx.y*blockDim.x+threadIdx.x; if(from>=64) return;
  const half_t* P=pl+(size_t)n*112*64; auto has=[&](int c){ return __half2float(P[c*64+from])>0.f; };
  int fr=from/8, ff=from%8; const uint8_t* occn=occ+(size_t)n*64;
  const int rd[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
  for(int side=0;side<2;side++){ int off=side*6; uint8_t* attn=(side==0?att_o:att_t)+(size_t)n*64;
    if(has(off+0)){ int dr=side==0?1:-1; for(int df=-1;df<=1;df+=2){int rr=fr+dr,cc=ff+df; if(rr>=0&&rr<8&&cc>=0&&cc<8) attn[rr*8+cc]=1;} }
    if(has(off+1)){ const int kd[8][2]={{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
      for(int k=0;k<8;k++){int rr=fr+kd[k][0],cc=ff+kd[k][1]; if(rr>=0&&rr<8&&cc>=0&&cc<8) attn[rr*8+cc]=1;} }
    if(has(off+5)) for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){ if(!dr&&!dc)continue; int rr=fr+dr,cc=ff+dc; if(rr>=0&&rr<8&&cc>=0&&cc<8) attn[rr*8+cc]=1; }
    bool bi=has(off+2),rk=has(off+3),qn=has(off+4);
    if(bi||rk||qn) for(int d=0;d<8;d++){ bool orth=d<4; if(!(orth?(rk||qn):(bi||qn))) continue; int rr=fr,cc=ff;
      for(int k=0;k<7;k++){ rr+=rd[d][0]; cc+=rd[d][1]; if(rr<0||rr>7||cc<0||cc>7)break; int t=rr*8+cc; attn[t]=1; if(occn[t])break; } } } }
// route id from piece + attacker bits
__global__ void k_route_id(const uint8_t* piece,const uint8_t* att_o,const uint8_t* att_t,int* route,int N){
  int n=blockIdx.x, s=blockIdx.y*blockDim.x+threadIdx.x; if(s>=64) return;
  size_t i=(size_t)n*64+s; int pc=piece[i],ao=att_o[i],at=att_t[i];
  route[i]= pc==0 ? ao+2*at : 4+(pc-1)+12*((pc<=6)?at:ao); }
// counting sort: histogram, then scatter rows into expert-contiguous order
__global__ void k_hist(const int* route,int* cnt,int R){ int r=blockIdx.x*blockDim.x+threadIdx.x; if(r<R) atomicAdd(&cnt[route[r]],1); }
__global__ void k_scatter_order(const int* route,int* cur,int* order,int R){ int r=blockIdx.x*blockDim.x+threadIdx.x; if(r<R){ int p=atomicAdd(&cur[route[r]],1); order[p]=r; } }

// ============================ the forward object ============================
struct HeroForward::Impl {
  HeroWeights w;               // host weights kept only for head-side host math
  cublasHandle_t cub;
  std::mutex mtx;              // lc0 calls ComputeBlocking from multiple search
                              // threads; one GPU -> serialize the forward (leaf
                              // collection still runs parallel on the CPU side)
  int d, L, H, hd, dff, E, ed, pd;
  float alpha;
  int capN = 0;                // batch the scratch is sized for (grows on demand)

  // ---- weights (device, uploaded once) ----
  half_t *pp0,*pp1,*emb,*eln,*eu,*edn,*efln,*zbuf;
  struct LW { half_t *qw,*kw,*vw,*ow,*l1g,*l2g,*up,*dn,*bias; };  // bias: (H,64,64) fp16
  std::vector<LW> lw;
  // head weights (device, uploaded once) + policy-promotion + gather map
  half_t *h_pe,*h_peb,*h_pq,*h_pqb,*h_pk,*h_pkb,*h_ve,*h_vq,*h_vk,*h_vv,*h_ppo;
  int* dGather=nullptr;        // 1858 policy move indices (uploaded on first Run)

  // ---- scratch (device, sized to capN) ----
  half_t *dPlanes,*dFlat,*pos128,*pos8192,*cat240,*emb_d,*e_out,*up_h,*dn_h,*x;
  half_t *qd,*kd,*vd,*po,*attn,*xa,*xs,*ffgh,*ys,*ffn,*dBias;
  // device head scratch
  half_t *tp,*qp,*kp,*scp,*promo,*tv,*qv,*kv,*vvh,*scv,*vout;
  float *d_pol,*d_wdl;
  int *dOrder;
  // device-route scratch
  uint8_t *dOcc,*dPiece,*dAtt_o,*dAtt_t; int *dRoute,*dCnt,*dCur;

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
      std::vector<float> bias_hhh((size_t)H*64*64, 0.f);   // free + alpha_mix . geo (once)
      for (int h=0;h<H;h++) for (int i=0;i<64;i++) for (int j=0;j<64;j++){
        double b=s.free[(size_t)h*64*64+i*64+j];
        for (int k=0;k<18;k++) b += (double)s.alpha[h*18+k]*geo[k][i*64+j];
        bias_hhh[((size_t)h*64+i)*64+j]=(float)b;
      }
      t.bias=up_f(bias_hhh);
    }
    // head weights once
    h_pe=up_f(w.pol_embed_w); h_peb=up_f(w.pol_embed_b);
    h_pq=up_f(w.pol_q_w); h_pqb=up_f(w.pol_q_b); h_pk=up_f(w.pol_k_w); h_pkb=up_f(w.pol_k_b);
    h_ve=up_f(w.val_embed_w); h_vq=up_f(w.val_q_w); h_vk=up_f(w.val_k_w); h_vv=up_f(w.val_v_w);
    h_ppo=up_f(w.pol_ppo_w);
    ensure(512);   // preallocate for the useful minibatch range (no mid-search realloc)
  }

  void ensure(int N) {                 // (re)allocate scratch for batch N
    if (N <= capN) return;
    if (capN) { for (half_t* p : {dPlanes,dFlat,pos128,pos8192,cat240,emb_d,e_out,up_h,dn_h,x,
                                   qd,kd,vd,po,attn,xa,xs,ffgh,ys,ffn,dBias,
                                   tp,qp,kp,scp,promo,tv,qv,kv,vvh,scv,vout}) cudaFree(p);
                cudaFree(dOrder); cudaFree(d_pol); cudaFree(d_wdl); }
    const size_t T=(size_t)N*64*d, R=(size_t)N*64;
    auto A=[&](half_t** p,size_t n){ CK(cudaMalloc(p,n*sizeof(half_t))); };
    A(&dPlanes,(size_t)N*112*64); A(&dFlat,(size_t)N*768); A(&pos128,(size_t)N*128);
    A(&pos8192,(size_t)N*8192); A(&cat240,R*240); A(&emb_d,T); A(&e_out,T);
    A(&up_h,R*ed); A(&dn_h,T); A(&x,T); A(&qd,T); A(&kd,T); A(&vd,T); A(&po,T);
    A(&attn,T); A(&xa,T); A(&xs,T); A(&ffgh,R*dff); A(&ys,T); A(&ffn,T);
    A(&dBias,(size_t)N*H*64*64);
    A(&tp,R*(size_t)pd); A(&qp,R*(size_t)pd); A(&kp,R*(size_t)pd); A(&scp,(size_t)N*4096);
    A(&promo,(size_t)N*8*4); A(&tv,R*(size_t)pd); A(&qv,R*(size_t)pd); A(&kv,R*(size_t)pd);
    A(&vvh,R*3); A(&scv,(size_t)N*4096); A(&vout,R*3);
    CK(cudaMalloc(&dOrder,R*sizeof(int)));
    CK(cudaMalloc(&d_pol,(size_t)N*1858*sizeof(float))); CK(cudaMalloc(&d_wdl,(size_t)N*3*sizeof(float)));
    if (capN) { cudaFree(dOcc);cudaFree(dPiece);cudaFree(dAtt_o);cudaFree(dAtt_t);cudaFree(dRoute); }
    CK(cudaMalloc(&dOcc,R)); CK(cudaMalloc(&dPiece,R)); CK(cudaMalloc(&dAtt_o,R)); CK(cudaMalloc(&dAtt_t,R));
    CK(cudaMalloc(&dRoute,R*sizeof(int)));
    if (!capN) { CK(cudaMalloc(&dCnt,(E+1)*sizeof(int))); CK(cudaMalloc(&dCur,E*sizeof(int))); }
    capN = N;
  }

  void route13(const float* pl, int N, std::vector<int>& route) {
    route.assign((size_t)N*64, 0);
    for (int n=0;n<N;n++) for (int s=0;s<64;s++) {
      int best=-1; float bv=0.f; bool any=false;
      for (int c=0;c<12;c++){ float v=pl[((size_t)n*112+c)*64+s]; if(v>0.f){any=true; if(best<0||v>bv){bv=v;best=c;}} }
      route[(size_t)n*64+s] = any ? best+1 : 0;
    }
  }

  // 28-class route (hero.py routes()/attackers()): empty squares -> control
  // (att_ours + 2*att_theirs, 0..3); occupied -> 4 + (piece-1) + 12*enemy_attacked.
  // Sliders see along a ray until (and including) the first occupied square.
  // Host-side per batch (cheap vs the trunk); planes[n,c,s] channels 0..5 ours
  // P/N/B/R/Q/K, 6..11 theirs.
  void route28(const float* pl, int N, std::vector<int>& route) {
    // precomputed target LISTS (<=8 per piece) instead of 64-wide masks; skip
    // empty from-squares. ~8x fewer inner iterations than the mask version.
    static bool init=false;
    static int kn_t[64][8],kn_n[64], kg_t[64][8],kg_n[64], pwo_t[64][2],pwo_n[64], pwt_t[64][2],pwt_n[64];
    static const int rays[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    if(!init){ init=true;
      for(int a=0;a<64;a++){ kn_n[a]=kg_n[a]=pwo_n[a]=pwt_n[a]=0;
        for(int t=0;t<64;t++){
          int ra=a/8,fa=a%8,rt=t/8,ft=t%8,dR=rt-ra,aF=abs(ft-fa),aR=abs(rt-ra);
          if((aR==2&&aF==1)||(aR==1&&aF==2)) kn_t[a][kn_n[a]++]=t;
          if(std::max(aR,aF)==1) kg_t[a][kg_n[a]++]=t;
          if(dR==1&&aF==1) pwo_t[a][pwo_n[a]++]=t;
          if(dR==-1&&aF==1) pwt_t[a][pwt_n[a]++]=t;
        } } }
    route.assign((size_t)N*64,0);
    auto Pn=[&](int n,int c,int s){ return pl[((size_t)n*112+c)*64+s]>0.f; };
    for(int n=0;n<N;n++){
      bool occ[64]; for(int s=0;s<64;s++){ occ[s]=false; for(int c=0;c<12;c++) if(Pn(n,c,s)){occ[s]=true;break;} }
      int att_ours[64]={0}, att_theirs[64]={0};
      for(int from=0;from<64;from++){ if(!occ[from]) continue;   // empty squares attack nothing
        for(int side=0;side<2;side++){ int off=side*6; int* att=side==0?att_ours:att_theirs;
          if(Pn(n,off+0,from)){ int* pt=side==0?pwo_t[from]:pwt_t[from]; int pn=side==0?pwo_n[from]:pwt_n[from]; for(int i=0;i<pn;i++) att[pt[i]]=1; }
          if(Pn(n,off+1,from)) for(int i=0;i<kn_n[from];i++) att[kn_t[from][i]]=1;
          if(Pn(n,off+5,from)) for(int i=0;i<kg_n[from];i++) att[kg_t[from][i]]=1;
          bool bi=Pn(n,off+2,from), rk=Pn(n,off+3,from), qn=Pn(n,off+4,from);
          if(bi||rk||qn) for(int d=0;d<8;d++){ bool orth=d<4; if(!(orth?(rk||qn):(bi||qn))) continue;
            int rr=from/8, ff=from%8;
            for(int k=0;k<7;k++){ rr+=rays[d][0]; ff+=rays[d][1]; if(rr<0||rr>7||ff<0||ff>7) break; int t=rr*8+ff; att[t]=1; if(occ[t]) break; } }
        } }
      for(int s=0;s<64;s++){
        int piece=0; float bv=0; for(int c=0;c<12;c++){ float v=pl[((size_t)n*112+c)*64+s]; if(v>0&&(piece==0||v>bv)){bv=v;piece=c+1;} }
        int rv; if(piece==0) rv=att_ours[s]+2*att_theirs[s];
        else { int enemy=(piece<=6)?att_theirs[s]:att_ours[s]; rv=4+(piece-1)+12*enemy; }
        route[(size_t)n*64+s]=rv;
      }
    }
  }
  void route(const float* pl,int N,std::vector<int>& r){ if(E<=13) route13(pl,N,r); else route28(pl,N,r); }

  // device gemm on a head weight into out_dev (rows x out), optional bias+mish
  void head_gemm(half_t* W, int in, int out, half_t* bias, bool mish,
                 half_t* xin_dev, int rows, half_t* out_dev) {
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,1.f,W,in,xin_dev,in,0.f,out_dev,out);
    if (bias) addBiasBatched<half_t>(out_dev,out_dev,bias,1,rows,out,mish?ACTIVATION_MISH:ACTIVATION_NONE,0);
    else if (mish) addBiasBatched<half_t>(out_dev,out_dev,zbuf,1,rows,out,ACTIVATION_MISH,0);
  }
};

// ---------------------------- public entry points ----------------------------
HeroForward::HeroForward(const HeroWeights& w) : p_(new Impl(w)) {}
HeroForward::~HeroForward() { delete p_; }

void HeroForward::Run(const float* planes_nchw, const float* flat12, int N,
                      const std::vector<int>& gather, float* policy_out, float* wdl_out) {
  Impl& I=*p_; std::lock_guard<std::mutex> lk(I.mtx); I.ensure(N); cublasHandle_t cub=I.cub;
  const int d=I.d,H=I.H,hd=I.hd,dff=I.dff,E=I.E,ed=I.ed,pd=I.pd; const float al=I.alpha;
  const size_t T=(size_t)N*64*d; const int R=N*64;

  // ---- optional phase profiling (HERO_PROFILE=1): stem/route/trunk/heads ----
  static const bool prof = getenv("HERO_PROFILE") != nullptr;
  static cudaEvent_t E0=0,Es=0,Er=0,Et=0,Eh=0;
  static double a_stem=0,a_route=0,a_trunk=0,a_heads=0; static long prof_calls=0;
  if (prof && !E0){ cudaEventCreate(&E0);cudaEventCreate(&Es);cudaEventCreate(&Er);cudaEventCreate(&Et);cudaEventCreate(&Eh); }
  if (prof) cudaEventRecord(E0,0);

  { // upload planes (fp32 -> fp16) via copyTypeConverted
    float* tmp; CK(cudaMalloc(&tmp,(size_t)N*112*64*sizeof(float)));
    CK(cudaMemcpy(tmp,planes_nchw,(size_t)N*112*64*sizeof(float),cudaMemcpyHostToDevice));
    copyTypeConverted(I.dPlanes,tmp,(int)((size_t)N*112*64),0); CK(cudaFree(tmp));
    CK(cudaMalloc(&tmp,(size_t)N*768*sizeof(float)));
    CK(cudaMemcpy(tmp,flat12,(size_t)N*768*sizeof(float),cudaMemcpyHostToDevice));
    copyTypeConverted(I.dFlat,tmp,(int)((size_t)N*768),0); CK(cudaFree(tmp));
  }

  // ---- stem ----
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,128,N,768,1.f,I.pp0,768,I.dFlat,768,0.f,I.pos128,128);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,8192,N,128,1.f,I.pp1,128,I.pos128,128,0.f,I.pos8192,8192);
  inputPreprocessForAttentionBody<half_t>(I.cat240,I.dPlanes,I.pos8192,N,112,128,true,0);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,240,1.f,I.emb,240,I.cat240,240,0.f,I.emb_d,d);
  LayerNorm<half_t>(R,d,I.e_out,I.emb_d,I.zbuf,(half_t*)nullptr,I.eln,I.zbuf,1e-3f,1.f,ACTIVATION_MISH,0);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,ed,R,d,1.f,I.eu,d,I.e_out,d,0.f,I.up_h,ed);
  addBiasBatched<half_t>(I.up_h,I.up_h,I.zbuf,1,R,ed,ACTIVATION_MISH,0);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,ed,1.f,I.edn,ed,I.up_h,ed,0.f,I.dn_h,d);
  LayerNorm<half_t>(R,d,I.x,I.dn_h,I.zbuf,I.e_out,I.efln,I.zbuf,1e-3f,al,ACTIVATION_NONE,0);

  if (prof) cudaEventRecord(Es,0);   // stem done
  // ---- route -> expert-contiguous order[] + host offsets[] ----
  std::vector<int> off(E+1,0);
  if (E<=13) {   // 13-class: cheap host route (unchanged)
    std::vector<int> route; I.route(planes_nchw,N,route);
    std::vector<int> order(R), cnt(E,0);
    for (int r=0;r<R;r++) cnt[route[r]]++;
    for (int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
    { std::vector<int> cur(off.begin(),off.end()-1); for(int r=0;r<R;r++) order[cur[route[r]]++]=r; }
    CK(cudaMemcpy(I.dOrder,order.data(),R*sizeof(int),cudaMemcpyHostToDevice));
  } else {       // 28-class: fully device-resident route + counting sort
    CK(cudaMemset(I.dAtt_o,0,R)); CK(cudaMemset(I.dAtt_t,0,R));
    k_occ_piece<<<dim3(N,1),64>>>(I.dPlanes,I.dOcc,I.dPiece,N);
    k_attackers<<<dim3(N,1),64>>>(I.dPlanes,I.dOcc,I.dAtt_o,I.dAtt_t,N);
    k_route_id<<<dim3(N,1),64>>>(I.dPiece,I.dAtt_o,I.dAtt_t,I.dRoute,N);
    CK(cudaMemset(I.dCnt,0,(E+1)*sizeof(int)));
    k_hist<<<(R+255)/256,256>>>(I.dRoute,I.dCnt,R);
    std::vector<int> cnt(E); CK(cudaMemcpy(cnt.data(),I.dCnt,E*sizeof(int),cudaMemcpyDeviceToHost));
    for (int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
    CK(cudaMemcpy(I.dCur,off.data(),E*sizeof(int),cudaMemcpyHostToDevice));   // cur = start offsets
    k_scatter_order<<<(R+255)/256,256>>>(I.dRoute,I.dCur,I.dOrder,R);
  }
  if (prof) cudaEventRecord(Er,0);   // route done

  // ---- trunk: 15 layers (fusedMHA + gather/scatter expert FFN) ----
  dim3 gd(R,(d+255)/256);
  const int HB=H*64*64;
  dim3 bg((HB+255)/256, N);
  for (int li=0; li<I.L; li++){
    auto& t=I.lw[li];
    k_bcast_bias<<<bg,256>>>(I.dBias, t.bias, HB);      // (H,64,64) -> (N,H,64,64), device only
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.qw,d,I.x,d,0.f,I.qd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.kw,d,I.x,d,0.f,I.kd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.vw,d,I.x,d,0.f,I.vd,d);
    fusedMHA<half_t>(I.po, I.qd, I.kd, I.vd, I.dBias, N, H, hd, 0);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,t.ow,d,I.po,d,0.f,I.attn,d);
    LayerNorm<half_t>(R,d,I.xa,I.attn,I.zbuf,I.x,t.l1g,I.zbuf,1e-3f,al,ACTIVATION_NONE,0);
    k_gather<<<gd,256>>>(I.xs,I.xa,I.dOrder,d);
    for (int e=0;e<E;e++){ int m=off[e+1]-off[e]; if(!m) continue;
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,m,d,1.f,t.up+(size_t)e*dff*d,d,I.xs+(size_t)off[e]*d,d,0.f,I.ffgh+(size_t)off[e]*dff,dff);
      addBiasBatched<half_t>(I.ffgh+(size_t)off[e]*dff,I.ffgh+(size_t)off[e]*dff,I.zbuf,1,m,dff,ACTIVATION_MISH,0);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,m,dff,1.f,t.dn+(size_t)e*d*dff,dff,I.ffgh+(size_t)off[e]*dff,dff,0.f,I.ys+(size_t)off[e]*d,d); }
    k_scatter<<<gd,256>>>(I.ffn,I.ys,I.dOrder,d);
    LayerNorm<half_t>(R,d,I.x,I.ffn,I.zbuf,I.xa,t.l2g,I.zbuf,1e-3f,al,ACTIVATION_NONE,0);  // x = next input
  }
  if (prof) cudaEventRecord(Et,0);   // trunk done

  // ---- heads (fully device-resident) ----
  if (!I.dGather) { CK(cudaMalloc(&I.dGather,1858*sizeof(int)));
    CK(cudaMemcpy(I.dGather,gather.data(),1858*sizeof(int),cudaMemcpyHostToDevice)); }
  const float sc = 1.f/sqrtf((float)pd), z=0.f, o=1.f;
  // policy: tp = mish(pol_embed(x)+b); qp,kp = pol_q/k(tp)+b; scp = qp.kp^T*sc; +promo; gather
  I.head_gemm(I.h_pe,d,pd,I.h_peb,true,I.x,R,I.tp);
  I.head_gemm(I.h_pq,pd,pd,I.h_pqb,false,I.tp,R,I.qp);
  I.head_gemm(I.h_pk,pd,pd,I.h_pkb,false,I.tp,R,I.kp);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,pd,&sc,I.kp,CUDA_R_16F,pd,64*pd,I.qp,CUDA_R_16F,pd,64*pd,&z,I.scp,CUDA_R_16F,64,64*64,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  { dim3 pg(N,8); k_promo<<<pg,4>>>(I.promo,I.kp,I.h_ppo,N,pd); }
  { dim3 gg(N,(1858+255)/256); k_pol_gather<<<gg,256>>>(I.d_pol,I.scp,I.promo,I.dGather,N); }
  // value: tv = mish(val_embed(x)); qv,kv,vv; softmax(qv.kv^T*sc) then .vv, mean over queries
  I.head_gemm(I.h_ve,d,pd,nullptr,true,I.x,R,I.tv);
  I.head_gemm(I.h_vq,pd,pd,nullptr,false,I.tv,R,I.qv);
  I.head_gemm(I.h_vk,pd,pd,nullptr,false,I.tv,R,I.kv);
  I.head_gemm(I.h_vv,pd,3,nullptr,false,I.tv,R,I.vvh);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,pd,&sc,I.kv,CUDA_R_16F,pd,64*pd,I.qv,CUDA_R_16F,pd,64*pd,&z,I.scv,CUDA_R_16F,64,64*64,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  Softmax<half_t>(N*64,64,I.scv,I.scv,(half_t*)nullptr,0);
  // vout(N,64,3) = scv(64x64) . vv(64x3), per position (see .cu notes for the layout)
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,3,64,64,&o,I.vvh,CUDA_R_16F,3,64*3,I.scv,CUDA_R_16F,64,64*64,&z,I.vout,CUDA_R_16F,3,64*3,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  k_wdl_mean<<<N,3>>>(I.d_wdl,I.vout,N);
  if (prof) cudaEventRecord(Eh,0);   // heads done
  CK(cudaDeviceSynchronize());
  if (prof) { float ms; long c=++prof_calls;
    cudaEventElapsedTime(&ms,E0,Es); a_stem+=ms;  cudaEventElapsedTime(&ms,Es,Er); a_route+=ms;
    cudaEventElapsedTime(&ms,Er,Et); a_trunk+=ms; cudaEventElapsedTime(&ms,Et,Eh); a_heads+=ms;
    if (c%50==0) fprintf(stderr,"HEROPROF N=%d/50-avg: stem %.2f | route %.2f | trunk %.2f | heads %.2f ms (sum %.1f)\n",
      N, a_stem/c, a_route/c, a_trunk/c, a_heads/c, (a_stem+a_route+a_trunk+a_heads)/c); }
  CK(cudaMemcpy(policy_out,I.d_pol,(size_t)N*1858*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(wdl_out,I.d_wdl,(size_t)N*3*sizeof(float),cudaMemcpyDeviceToHost));
}

void HeroForward::DebugRoute(const float* planes_nchw, int N, std::vector<int>& out) {
  Impl& I=*p_; std::lock_guard<std::mutex> lk(I.mtx); I.ensure(N); const int R=N*64;
  float* tmp; CK(cudaMalloc(&tmp,(size_t)N*112*64*sizeof(float)));
  CK(cudaMemcpy(tmp,planes_nchw,(size_t)N*112*64*sizeof(float),cudaMemcpyHostToDevice));
  copyTypeConverted(I.dPlanes,tmp,(int)((size_t)N*112*64),0); CK(cudaFree(tmp));
  CK(cudaMemset(I.dAtt_o,0,R)); CK(cudaMemset(I.dAtt_t,0,R));
  k_occ_piece<<<dim3(N,1),64>>>(I.dPlanes,I.dOcc,I.dPiece,N);
  k_attackers<<<dim3(N,1),64>>>(I.dPlanes,I.dOcc,I.dAtt_o,I.dAtt_t,N);
  k_route_id<<<dim3(N,1),64>>>(I.dPiece,I.dAtt_o,I.dAtt_t,I.dRoute,N);
  CK(cudaDeviceSynchronize());
  out.resize(R); CK(cudaMemcpy(out.data(),I.dRoute,R*sizeof(int),cudaMemcpyDeviceToHost));
}

}  // namespace hero
}  // namespace lczero
