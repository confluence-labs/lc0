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
#include <map>
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

#include "neural/hero/hero_qkv_band.h"   // lc0bench 0008

namespace lczero {
namespace hero {

// lc0bench 0007: the fused residual-add + LayerNorm (hero_ln.cu). Same
// arithmetic as common_kernels' LayerNorm with bias/beta zero and act NONE:
// out = gamma * (input*alpha + skip - mean) / sqrt(var + eps), stats in fp32.
void heroAddLN(half* o, const half* x, const half* skip, const half* gam,
               int N, int C, float eps, float alpha, cudaStream_t stream);
bool heroAddLNSupported(int C);

// lc0bench 0006: the grouped expert FFN (hero_gffn.cu). Rows are class-SORTED and
// the row tile's class comes from a device-side tile->class map; mish is fused in
// the up epilogue and the scatter through dOrder is the down epilogue.
bool heroGFFNSupported(int d, int dff);
int  heroGFFNPad(int dff);
int  heroGFFNTileRows();
int  heroGFFNPadRows();
void heroGFFNMap(const int* cnt, int* offs, int* cur, int* tmap, int E, int ntmax,
                 cudaStream_t stream);
void heroGFFNUp(half* h, const half* xs, const half* upw, const int* tmap,
                const int* offs, const int* row, int R, int dp, int d, int ntiles,
                cudaStream_t stream);
void heroGFFNDown(half* out, const half* h, const half* dnw, const int* tmap,
                  const int* offs, const int* row, int R, int d, int dff, int ldh,
                  int ntiles, cudaStream_t stream);
void heroGFFNCompare(const half* a, const half* b, int n, float* scratch, float* host2);

// lc0bench 0005: the hand-written band (hero_band.cu). Same contract as
// fusedMHA: q/k/v/o are [N*64, H*hd] fp16 row-major with head-major columns,
// bias is [H,64,64] fp16 broadcast over the batch, scale 1/sqrt(hd).
void heroBand(half* o, const half* q, const half* k, const half* v,
              const half* bias, int N, int H, int hd, cudaStream_t stream);

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
// quantize [outC,inC] row-major fp32 weights -> int8 device (per-out-channel) + inv[outC]=amax/127
static void quant_chan(const float* w, size_t outC, int inC, int8_t** d_i8, float** d_inv) {
  std::vector<int8_t> q(outC*inC); std::vector<float> inv(outC);
  for (size_t o=0;o<outC;o++){ float mx=0; const float* wr=w+o*inC; for(int j=0;j<inC;j++) mx=fmaxf(mx,fabsf(wr[j]));
    float sc=mx>0?127.f/mx:0.f; inv[o]=mx>0?mx/127.f:0.f;
    for(int j=0;j<inC;j++){ int v=(int)lrintf(wr[j]*sc); q[o*inC+j]=(int8_t)(v>127?127:v<-127?-127:v); } }
  CK(cudaMalloc(d_i8,outC*inC)); CK(cudaMemcpy(*d_i8,q.data(),outC*inC,cudaMemcpyHostToDevice));
  CK(cudaMalloc(d_inv,outC*sizeof(float))); CK(cudaMemcpy(*d_inv,inv.data(),outC*sizeof(float),cudaMemcpyHostToDevice));
}
// int8 gemm: C[m,n] int32 = op_T(A_i8[k,m]) * op_N(B_i8[k,n]), int32 accumulate (TN format)
static void gemm_i8(cublasHandle_t h,int m,int n,int k,const int8_t* A,int lda,const int8_t* B,int ldb,int32_t* C,int ldc){
  int32_t a=1,b=0;
  CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&a,A,CUDA_R_8I,lda,B,CUDA_R_8I,ldb,&b,C,CUDA_R_32I,ldc,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
}
// SmoothQuant: migrate per-input-channel scale s = amax(act)^.5 / amax(w)^.5 into the
// weights (W'=W*s), quantize per-out-channel, and emit the 1/s activation prescale.
static void sq_quant(const float* w, size_t outC, int inC, const float* Achan,
                     int8_t** d_i8, float** d_inv, float** d_sqinv){
  std::vector<float> Wcol(inC,0.f);
  for(size_t o=0;o<outC;o++){ const float* wr=w+o*inC; for(int j=0;j<inC;j++) Wcol[j]=fmaxf(Wcol[j],fabsf(wr[j])); }
  std::vector<float> s(inC), sqinv(inC);
  for(int j=0;j<inC;j++){ float a=Achan[j],wc=Wcol[j]; float sj=(a>0&&wc>0)?sqrtf(a)/sqrtf(wc):1.f;
    sj=fmaxf(1e-2f,fminf(1e2f,sj)); s[j]=sj; sqinv[j]=1.f/sj; }
  std::vector<float> Wp(outC*inC);
  for(size_t o=0;o<outC;o++)for(int j=0;j<inC;j++) Wp[o*inC+j]=w[o*inC+j]*s[j];
  quant_chan(Wp.data(),outC,inC,d_i8,d_inv);
  CK(cudaMalloc(d_sqinv,inC*sizeof(float))); CK(cudaMemcpy(*d_sqinv,sqinv.data(),inC*sizeof(float),cudaMemcpyHostToDevice));
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

// device kernels: FFN gather/scatter (rows pre-sorted by expert). Vectorized to
// 16-byte (int4 = 8xhalf) loads — d is a multiple of 8 for all hero configs, so
// every row offset is 16-byte aligned. Bit-identical to per-half, ~8x bandwidth.
__global__ void k_gather(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x, c8=blockIdx.y*blockDim.x+threadIdx.x, d8=d>>3;
  if(c8<d8) ((int4*)o)[(size_t)r*d8+c8]=((const int4*)in)[(size_t)idx[r]*d8+c8]; }
__global__ void k_scatter(half_t* o,const half_t* in,const int* idx,int d){
  int r=blockIdx.x, c8=blockIdx.y*blockDim.x+threadIdx.x, d8=d>>3;
  if(c8<d8) ((int4*)o)[(size_t)idx[r]*d8+c8]=((const int4*)in)[(size_t)r*d8+c8]; }
// broadcast per-head bias (H,64,64) -> (N,H,64,64) for the bias input (batch-independent)
__global__ void k_bcast_bias(half_t* o,const half_t* b,int HB){  // HB = H*64*64
  int n=blockIdx.y, x=blockIdx.x*blockDim.x+threadIdx.x; if(x<HB) o[(size_t)n*HB+x]=b[x]; }
// (N,64,H*hd) interleaved <-> (N,H,64,hd) per-head layout, for the non-fused
// attention path (hd not in {32,64,128} -> fusedMHA can't take it; e.g. run8d hd=5)
__global__ void k_toheads(half_t* o,const half_t* in,int H,int hd){
  int n=blockIdx.x,s=blockIdx.y,h=blockIdx.z,e=threadIdx.x; if(e<hd) o[(((size_t)n*H+h)*64+s)*hd+e]=in[((size_t)n*64+s)*(H*hd)+h*hd+e]; }
__global__ void k_fromheads(half_t* o,const half_t* in,int H,int hd){
  int n=blockIdx.x,s=blockIdx.y,h=blockIdx.z,e=threadIdx.x; if(e<hd) o[((size_t)n*64+s)*(H*hd)+h*hd+e]=in[(((size_t)n*H+h)*64+s)*hd+e]; }
// softmax over 64 keys + per-(head,query) STATIC bias broadcast over batch — avoids
// materializing (N,H,64,64) dBias every layer (was ~8GB/fwd write+read for run8d H=128).
// rows = N*H*64, one block/row, 64 threads; bias is (H,64,64), biasrow = row % (H*64).
__global__ void k_softmax_bias(half_t* sc,const half_t* bias,int rows,int HB){
  int r=blockIdx.x; if(r>=rows) return; int j=threadIdx.x;
  __shared__ float sm[64];
  float v=__half2float(sc[(size_t)r*64+j]) + __half2float(bias[(size_t)(r%HB)*64+j]);
  sm[j]=v; __syncthreads();
  float mx=-1e30f; for(int k=0;k<64;k++) mx=fmaxf(mx,sm[k]);
  float e=__expf(v-mx); sm[j]=e; __syncthreads();
  float s=0; for(int k=0;k<64;k++) s+=sm[k];
  sc[(size_t)r*64+j]=__float2half(e/s); }
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

// lc0bench 0009: the square-major first-12 flat the positional preproc consumes,
// straight off the packed planes. The host loop this replaces wrote
// flat12[n*768 + s*12 + c] = value(c) for every set bit s of plane c < 12 into a
// zero-initialised buffer; every (n,s,c) is written here, so it needs no memset.
__global__ void k_flat12(half_t* o,const uint64_t* masks,const half_t* vals,int N){
  int n=blockIdx.x, s=threadIdx.x;
  for(int c=0;c<12;c++){ uint64_t m=masks[(size_t)n*112+c];
    o[(size_t)n*768+s*12+c] = ((m>>s)&1ull) ? vals[(size_t)n*112+c] : (half_t)0.f; } }

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

// ---- int8 quantization helpers (INC int8-A: FFN gemms) ----
// per-row dynamic quant of a [rows,K] fp16 matrix -> int8 + per-row inv-scale (amax/127).
// `pre` (nullptr or per-channel): SmoothQuant activation smoothing x[j] *= pre[j] first.
__global__ void k_quant_rows(const half_t* x, int8_t* xq, float* ainv, int rows, int K, const float* pre){
  int r=blockIdx.x; if(r>=rows) return;
  const half_t* xr=x+(size_t)r*K; int8_t* qr=xq+(size_t)r*K;
  __shared__ float sm[256]; float amax=0;
  for(int j=threadIdx.x;j<K;j+=blockDim.x){ float v=__half2float(xr[j])*(pre?pre[j]:1.f); amax=fmaxf(amax,fabsf(v)); }
  sm[threadIdx.x]=amax; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]); __syncthreads(); }
  float mx=sm[0]; float sc = mx>0? 127.f/mx : 0.f;
  if(threadIdx.x==0) ainv[r] = mx>0? mx/127.f : 0.f;
  for(int j=threadIdx.x;j<K;j+=blockDim.x){ float v=__half2float(xr[j])*(pre?pre[j]:1.f); int q=__float2int_rn(v*sc); qr[j]=(int8_t)max(-127,min(127,q)); }
}
// calibration: accumulate per-channel abs-max over rows (SmoothQuant activation stats)
__global__ void k_chan_absmax(const half_t* x, float* out, int rows, int K){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=K) return;
  float m=out[j]; for(int r=0;r<rows;r++) m=fmaxf(m,fabsf(__half2float(x[(size_t)r*K+j]))); out[j]=m;
}
// FUSED dequant(int32)->mish->per-row requant(int8): reads the up-gemm int32 accum once,
// writes int8 once (halves the fp16 round-trip of separate dequant+requant). `pre` = SQ 1/s.
__global__ void k_dq_mish_q(const int32_t* c, int8_t* q, float* rowinv, const float* wcol_inv,
                            const float* ainv, const float* pre, int K, int rows){
  extern __shared__ float sm[];   // K floats: the mished row
  int r=blockIdx.x; if(r>=rows) return; const int32_t* cr=c+(size_t)r*K; float ar=ainv[r];
  float amax=0;
  for(int j=threadIdx.x;j<K;j+=blockDim.x){
    float v=(float)cr[j]*wcol_inv[j]*ar; float sp=v>20.f?v:logf(1.f+expf(v)); v=v*tanhf(sp);
    if(pre) v*=pre[j]; sm[j]=v; amax=fmaxf(amax,fabsf(v)); }
  __shared__ float red[256]; red[threadIdx.x]=amax; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+s]); __syncthreads(); }
  float mx=red[0], sc=mx>0?127.f/mx:0.f; if(threadIdx.x==0) rowinv[r]=mx>0?mx/127.f:0.f;
  int8_t* qr=q+(size_t)r*K;
  for(int j=threadIdx.x;j<K;j+=blockDim.x){ int qq=__float2int_rn(sm[j]*sc); qr[j]=(int8_t)max(-127,min(127,qq)); }
}
// dequant int32 gemm output C[out,rows] (col-major, ld=out) with per-out-channel wcol_inv
// and per-row ainv; optional mish. writes fp16 in the same [out,rows] layout.
__global__ void k_dequant(const int32_t* c, half_t* out, const float* wcol_inv, const float* ainv, int outC, int rows, int mish){
  int o=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y; if(o>=outC) return;
  size_t idx=(size_t)r*outC+o; float v=(float)c[idx]*wcol_inv[o]*ainv[r];
  if(mish){ float sp = v>20.f? v : logf(1.f+expf(v)); v = v*tanhf(sp); }
  out[idx]=__float2half(v);
}

// ============================ the forward object ============================
struct HeroForward::Impl {
  HeroWeights w;               // host weights kept only for head-side host math
  cublasHandle_t cub;
  static const int NS = 8;    // streams for concurrent expert FFN (experts are
  cudaStream_t rs = nullptr;  // lc0bench 0010: Run()'s one stream (the legacy one is uncapturable)
  std::map<int,cudaGraphExec_t> graphs; std::map<int,int> seen;   // lc0bench 0010: one graph per N
  bool nograph = getenv("HERO_NO_GRAPH") != nullptr;
  void drop_graphs(){ for (auto& g : graphs) cudaGraphExecDestroy(g.second); graphs.clear(); seen.clear(); }
  cudaStream_t streams[NS];   // mutually independent; run them in parallel
  cudaEvent_t ev_gather, ev_done[NS];
  std::mutex mtx;             // lc0 calls ComputeBlocking from multiple search
                             // threads; one GPU -> serialize the forward (leaf
                             // collection still runs parallel on the CPU side)
  int d, L, H, hd, dff, E, ed, pd, bank;   // bank = H*hd (attention width); may != d (bankdeep)
  bool fused;                              // fusedMHA supports hd in {32,64,128}; else non-fused path (run8d hd=5)
  bool qkvb=false;                         // lc0bench 0008: hero_qkv_band.cu instead of 3 GEMMs + the band
  bool qkvbw=false;                        // lc0bench 0011: its warp-specialized entry point
  bool fastln;                             // lc0bench 0007: hero_ln.cu instead of LayerNorm<half_t>
  bool band;                               // lc0bench 0005: hero_band.cu instead of fusedMHA
  float alpha;
  int capN = 0;                // batch the scratch is sized for (grows on demand)
  // lc0bench 0012: per-call plane slots. The upload runs on upst[k] OUTSIDE mtx; the forward waits on
  // upDone[k] and records slotFree[k] when it has finished reading the slot.
  static const int NSLOT = 4;
  const bool asyncup = getenv("HERO_NO_ASYNCUP") == nullptr;
  std::mutex slotmtx[NSLOT], slotpick; int slotrr = 0, slotcap = 0;
  cudaStream_t upst[NSLOT] = {};
  cudaEvent_t upDone[NSLOT] = {}, slotFree[NSLOT] = {};
  half_t *sPlanes[NSLOT] = {}, *sFlat[NSLOT] = {}, *sVal[NSLOT] = {};
  uint64_t *sMask[NSLOT] = {}; float *sStgV[NSLOT] = {};
  // lc0bench 0006: the grouped expert FFN. dp is the up-GEMM's padded column count
  // (== dff whenever dff is already a multiple of 128, which costs hero3 nothing).
  bool gffn = false; int dp = 0;
  int *dOffs=nullptr, *dTmap=nullptr;   // [E+1] class row offsets; [2*ntmax] tile->class
  const bool gffn_check = getenv("HERO_GFFN_CHECK") != nullptr;
  const bool no_agat = getenv("HERO_NO_AGAT") != nullptr;   // A/B the A-load gather alone
  half_t *ffnref=nullptr; float *cmpd=nullptr;   // HERO_GFFN_CHECK only
  const bool int8 = getenv("HERO_INT8") != nullptr;   // INC int8-A: int8 FFN gemms
  const bool calib = getenv("HERO_CALIB") != nullptr; // collect per-channel act stats
  const bool sq = getenv("HERO_SQ") != nullptr;       // apply SmoothQuant (needs HERO_SQ_FILE)
  const bool cutlass_ffn = getenv("HERO_CUTLASS_FFN") != nullptr;  // fuse mish into up-gemm (CUTLASS)
  float *calib_up=nullptr,*calib_dn=nullptr;          // [L*d],[L*dff] activation abs-max

  // ---- weights (device, uploaded once) ----
  half_t *pp0,*pp1,*emb,*eln,*eu,*edn,*efln,*zbuf;
  struct LW { half_t *qw,*kw,*vw,*ow,*l1g,*l2g,*up,*dn,*bias;   // bias: (H,64,64) fp16
              half_t *upp=nullptr;                              // lc0bench 0006: [E,dp,d] up weights
              int8_t *up_i8=nullptr,*dn_i8=nullptr; float *up_inv=nullptr,*dn_inv=nullptr;
              float *sq_up_inv=nullptr,*sq_dn_inv=nullptr; };   // SmoothQuant 1/s prescales
  std::vector<LW> lw;
  // head weights (device, uploaded once) + policy-promotion + gather map
  half_t *h_pe,*h_peb,*h_pq,*h_pqb,*h_pk,*h_pkb,*h_ve,*h_vq,*h_vk,*h_vv,*h_ppo;
  half_t *h_veb=nullptr,*h_vqb=nullptr,*h_vkb=nullptr;   // lc0bench: optional value biases
  int* dGather=nullptr;        // 1858 policy move indices (uploaded on first Run)

  // ---- scratch (device, sized to capN) ----
  half_t *dPlanes,*dFlat,*pos128,*pos8192,*cat240,*emb_d,*e_out,*up_h,*dn_h,*x;
  float *stgP=nullptr,*stgF=nullptr;   // mfu60: fp32 H2D staging, sized in ensure()
  uint64_t* dMask=nullptr; half_t* dVal=nullptr; float* stgV=nullptr;   // lc0bench 0009
  half_t *qd,*kd,*vd,*po,*attn,*xa,*xs,*ffgh,*ys,*ffn;
  half_t *qt=nullptr,*kt=nullptr,*vt=nullptr,*sca=nullptr,*ctx=nullptr;  // non-fused attn scratch
  // device head scratch
  half_t *tp,*qp,*kp,*scp,*promo,*tv,*qv,*kv,*vvh,*scv,*vout;
  float *d_pol,*d_wdl;
  int *dOrder;
  // device-route scratch
  uint8_t *dOcc,*dPiece,*dAtt_o,*dAtt_t; int *dRoute,*dCnt,*dCur;
  // int8 FFN scratch (allocated only when int8)
  int8_t *xs_i8=nullptr,*gh_i8=nullptr; int32_t *ffgh_i32=nullptr,*ys_i32=nullptr; float *xrow_inv=nullptr,*ghrow_inv=nullptr;

  int device = 0;              // GPU id (multi-GPU: multiplexing gives gpu=0/1/...)
  Impl(const HeroWeights& wt, int gpu) : w(wt), device(gpu) {
    CK(cudaSetDevice(device));
    d=w.d; L=w.layers; H=w.heads; hd=w.hd; dff=w.dff; E=w.classes; ed=w.embed_dff; pd=w.pol_d;
    bank=H*hd;   // attention bank width (q/k/v/out + scratch use this, NOT d — bankdeep has bank>>d)
    fused = (hd==32||hd==64||hd==128);   // else CUTLASS fmha misaligns; use the batched-gemm path
    // lc0bench 0007: shape-generic (d % 16 == 0, d <= 2048). HERO_NO_FASTLN=1 = A/B.
    fastln = heroAddLNSupported(d) && getenv("HERO_NO_FASTLN")==nullptr;
    // lc0bench 0005: deep-bank nets only (heroD4: bank 4096 != d 640), so hero3
    // (square, bank == d) stays on fusedMHA and bit-identical. HERO_NO_BAND=1 = A/B.
    band = (hd==32 && getenv("HERO_NO_BAND")==nullptr);   // square banks too (hero3)
    // lc0bench 0008: the fused q|k|v+band program. hd 32 only (one mma k-step pair),
    // and it implies `fused`, so the batched-gemm attention path is never reached.
    qkvb = (hd==32 && getenv("HERO_NO_QKVB")==nullptr);
    qkvbw = qkvb && getenv("HERO_NO_WS")==nullptr;   // lc0bench 0011
    alpha = powf(2.f*L, -0.25f);
    // lc0bench 0006: shape-generic; steps aside for int8 / CUTLASS-FFN / calibration.
    gffn = heroGFFNSupported(d,dff) && !int8 && !cutlass_ffn && !calib && E>1
           && getenv("HERO_NO_GFFN")==nullptr;
    dp = gffn ? heroGFFNPad(dff) : dff;
    CK(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking));   // lc0bench 0010
    for (int k=0;k<NSLOT;k++){   // lc0bench 0012
      CK(cudaStreamCreateWithFlags(&upst[k], cudaStreamNonBlocking));
      CK(cudaEventCreateWithFlags(&upDone[k], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&slotFree[k], cudaEventDisableTiming));
      CK(cudaEventRecord(slotFree[k], rs));   // a slot starts free
    }
    CB(cublasCreate(&cub)); CB(cublasSetMathMode(cub, CUBLAS_TENSOR_OP_MATH)); CB(cublasSetStream(cub, rs));
    for (int s=0;s<NS;s++){ cudaStreamCreate(&streams[s]); cudaEventCreateWithFlags(&ev_done[s],cudaEventDisableTiming); }
    cudaEventCreateWithFlags(&ev_gather,cudaEventDisableTiming);
    pp0=up_f(w.preproc0_w); pp1=up_f(w.preproc1_w); emb=up_f(w.embed_w); eln=up_f(w.embed_ln_g);
    eu=up_f(w.embed_up_w); edn=up_f(w.embed_down_w); efln=up_f(w.embed_ffn_ln_g);
    std::vector<float> zeros((d>dff?d:dff),0.f); zbuf=up_f(zeros);
    auto geo = geo_basis();
    // SmoothQuant: load per-input-channel activation abs-max (calibration) once
    std::vector<float> cal_up, cal_dn;
    if (sq) {
      const char* cf=getenv("HERO_SQ_FILE"); FILE* f=cf?fopen(cf,"rb"):nullptr;
      if(!f){ fprintf(stderr,"HERO_SQ set but HERO_SQ_FILE missing\n"); exit(1); }
      cal_up.resize((size_t)L*d); cal_dn.resize((size_t)L*dff);
      if(fread(cal_up.data(),sizeof(float),cal_up.size(),f)!=cal_up.size()||
         fread(cal_dn.data(),sizeof(float),cal_dn.size(),f)!=cal_dn.size()){ fprintf(stderr,"short calib file\n"); exit(1); }
      fclose(f);
    }
    if (calib) { CK(cudaMalloc(&calib_up,(size_t)L*d*sizeof(float))); CK(cudaMemset(calib_up,0,(size_t)L*d*sizeof(float)));
                 CK(cudaMalloc(&calib_dn,(size_t)L*dff*sizeof(float))); CK(cudaMemset(calib_dn,0,(size_t)L*dff*sizeof(float))); }
    lw.resize(L);
    for (int li=0; li<L; li++) {
      auto& s=w.layer[li]; auto& t=lw[li];
      t.qw=up_f(s.q_w); t.kw=up_f(s.k_w); t.vw=up_f(s.v_w); t.ow=up_f(s.out_w);
      t.l1g=up_f(s.ln1_g); t.l2g=up_f(s.ln2_g);
      if (int8) {  // per-output-channel int8 for the whole expert bank; free the fp16 copies
        if (sq) {
          sq_quant(s.ffn_up.data(),  (size_t)E*dff,d, cal_up.data()+(size_t)li*d,   &t.up_i8,&t.up_inv,&t.sq_up_inv);
          sq_quant(s.ffn_down.data(),(size_t)E*d, dff,cal_dn.data()+(size_t)li*dff, &t.dn_i8,&t.dn_inv,&t.sq_dn_inv);
        } else {
          quant_chan(s.ffn_up.data(),   (size_t)E*dff, d,   &t.up_i8, &t.up_inv);
          quant_chan(s.ffn_down.data(), (size_t)E*d,   dff, &t.dn_i8, &t.dn_inv);
        }
        t.up=t.dn=nullptr;
      } else { t.up=up_f(s.ffn_up); t.dn=up_f(s.ffn_down);
        if (gffn && dp!=dff) {   // lc0bench 0006: [E,dp,d], pad columns zero -> mish(0)=0
          std::vector<float> pad((size_t)E*dp*d, 0.f);
          for (int e=0;e<E;e++) for (size_t i=0;i<(size_t)dff*d;i++)
            pad[(size_t)e*dp*d+i] = s.ffn_up[(size_t)e*dff*d+i];
          t.upp=up_f(pad);
        } else t.upp=t.up; }
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
    if (!w.val_embed_b.empty()) h_veb=up_f(w.val_embed_b);
    if (!w.val_q_b.empty())     h_vqb=up_f(w.val_q_b);
    if (!w.val_k_b.empty())     h_vkb=up_f(w.val_k_b);
    h_ppo=up_f(w.pol_ppo_w);
    ensure(512);   // preallocate for the useful minibatch range (no mid-search realloc)
  }

  void ensure(int N) {                 // (re)allocate scratch for batch N
    if (N <= capN) return;
    drop_graphs();   // lc0bench 0010: a graph bakes in the pointers it was captured with
    if (capN) { for (half_t* p : {dPlanes,dFlat,pos128,pos8192,cat240,emb_d,e_out,up_h,dn_h,x,
                                   qd,kd,vd,po,attn,xa,xs,ffgh,ys,ffn,
                                   tp,qp,kp,scp,promo,tv,qv,kv,vvh,scv,vout}) cudaFree(p);
                cudaFree(dOrder); cudaFree(d_pol); cudaFree(d_wdl); if (dTmap) cudaFree(dTmap); }
    const size_t T=(size_t)N*64*d, R=(size_t)N*64, Tbank=(size_t)R*bank;
    auto A=[&](half_t** p,size_t n){ CK(cudaMalloc(p,n*sizeof(half_t))); };
    A(&dPlanes,(size_t)N*112*64); A(&dFlat,(size_t)N*768); A(&pos128,(size_t)N*128);
    A(&pos8192,(size_t)N*8192); A(&cat240,R*240); A(&emb_d,T); A(&e_out,T);
    // lc0bench 0008: the fused program never materialises the three bank buffers
    // (-603 MB at N=384 on heroD4, -804 at 512). cudaFree(nullptr) is a no-op, so
    // the free list at the top of ensure() needs no change.
    A(&up_h,R*ed); A(&dn_h,T); A(&x,T);
    if (qkvb) { qd=kd=vd=nullptr; } else { A(&qd,Tbank); A(&kd,Tbank); A(&vd,Tbank); }
    A(&po,Tbank);
    if (!fused) {  // non-fused attention scratch (per-head layout + scores + broadcast bias)
      if (capN) { for (half_t* p : {qt,kt,vt,sca,ctx}) cudaFree(p); }
      A(&qt,Tbank); A(&kt,Tbank); A(&vt,Tbank); A(&ctx,Tbank);
      A(&sca,(size_t)N*H*64*64);   // scores; bias applied in-kernel (no dBias materialization)
    }
    // lc0bench 0006: a partial class tile reads up to BM-1 rows past its class (its
    // stores are masked), so both grouped A buffers carry spare rows; ffgh is dp wide.
    const size_t gpad = gffn ? (size_t)heroGFFNPadRows() : 0;
    A(&attn,T); A(&xa,T); A(&xs,T+gpad*d); A(&ffgh,(R+gpad)*(size_t)dp); A(&ys,T); A(&ffn,T);
    if (gffn) { CK(cudaMemset(xs+T,0,gpad*d*sizeof(half_t)));
               CK(cudaMemset(ffgh+R*(size_t)dp,0,gpad*(size_t)dp*sizeof(half_t))); }
    if (gffn_check) { if (capN) { cudaFree(ffnref); } A(&ffnref,T); }
    A(&tp,R*(size_t)pd); A(&qp,R*(size_t)pd); A(&kp,R*(size_t)pd); A(&scp,(size_t)N*4096);
    A(&promo,(size_t)N*8*4); A(&tv,R*(size_t)pd); A(&qv,R*(size_t)pd); A(&kv,R*(size_t)pd);
    A(&vvh,R*3); A(&scv,(size_t)N*4096); A(&vout,R*3);
    if (capN) { cudaFree(stgP); cudaFree(stgF); cudaFree(dMask); cudaFree(dVal); cudaFree(stgV); }
    CK(cudaMalloc(&stgP,(size_t)N*112*64*sizeof(float)));
    CK(cudaMalloc(&stgF,(size_t)N*768*sizeof(float)));
    CK(cudaMalloc(&dMask,(size_t)N*112*sizeof(uint64_t)));   // lc0bench 0009: 0.34 MB at N=384
    CK(cudaMalloc(&dVal,(size_t)N*112*sizeof(half_t)));
    CK(cudaMalloc(&stgV,(size_t)N*112*sizeof(float)));
    // lc0bench 0012: the slots are allocated ONCE, at the first ensure() (the ctor's, N=512), and never
    // freed. An upload runs outside mtx, so a realloc here would pull a buffer out from under an
    // in-flight call -- and ensure() cannot take the slot mutexes, because the caller may already hold
    // one. A batch bigger than the slots simply takes the old in-mutex path (`N <= slotcap` below).
    if (!slotcap) for (int k=0;k<NSLOT;k++){
      CK(cudaMalloc(&sPlanes[k],(size_t)N*112*64*sizeof(half_t)));
      CK(cudaMalloc(&sFlat[k],(size_t)N*768*sizeof(half_t)));
      CK(cudaMalloc(&sVal[k],(size_t)N*112*sizeof(half_t)));
      CK(cudaMalloc(&sMask[k],(size_t)N*112*sizeof(uint64_t)));
      CK(cudaMalloc(&sStgV[k],(size_t)N*112*sizeof(float)));
      slotcap = N;
    }
    CK(cudaMalloc(&dOrder,R*sizeof(int)));
    if (gffn) CK(cudaMalloc(&dTmap,2*((size_t)R/heroGFFNTileRows()+E+1)*sizeof(int)));   // lc0bench 0006
    CK(cudaMalloc(&d_pol,(size_t)N*1858*sizeof(float))); CK(cudaMalloc(&d_wdl,(size_t)N*3*sizeof(float)));
    if (capN) { cudaFree(dOcc);cudaFree(dPiece);cudaFree(dAtt_o);cudaFree(dAtt_t);cudaFree(dRoute); }
    CK(cudaMalloc(&dOcc,R)); CK(cudaMalloc(&dPiece,R)); CK(cudaMalloc(&dAtt_o,R)); CK(cudaMalloc(&dAtt_t,R));
    CK(cudaMalloc(&dRoute,R*sizeof(int)));
    if (!capN) { CK(cudaMalloc(&dCnt,(E+1)*sizeof(int))); CK(cudaMalloc(&dCur,E*sizeof(int)));
      if (gffn) CK(cudaMalloc(&dOffs,(E+1)*sizeof(int)));                      // lc0bench 0006
      if (gffn_check) CK(cudaMalloc(&cmpd,2*sizeof(float))); }
    if (int8) { if(capN){ cudaFree(xs_i8);cudaFree(gh_i8);cudaFree(ffgh_i32);cudaFree(ys_i32);cudaFree(xrow_inv);cudaFree(ghrow_inv); }
      CK(cudaMalloc(&xs_i8,R*(size_t)d)); CK(cudaMalloc(&gh_i8,R*(size_t)dff));
      CK(cudaMalloc(&ffgh_i32,R*(size_t)dff*sizeof(int32_t))); CK(cudaMalloc(&ys_i32,R*(size_t)d*sizeof(int32_t)));
      CK(cudaMalloc(&xrow_inv,R*sizeof(float))); CK(cudaMalloc(&ghrow_inv,R*sizeof(float))); }
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
    if (bias) addBiasBatched<half_t>(out_dev,out_dev,bias,1,rows,out,mish?ACTIVATION_MISH:ACTIVATION_NONE,rs);
    else if (mish) addBiasBatched<half_t>(out_dev,out_dev,zbuf,1,rows,out,ACTIVATION_MISH,rs);
  }
};

// ---------------------------- public entry points ----------------------------
HeroForward::HeroForward(const HeroWeights& w, int gpu) : p_(new Impl(w, gpu)) {}
HeroForward::~HeroForward() { delete p_; }

void HeroForward::Run(const float* planes_nchw, const float* flat12, int N,
                      const std::vector<int>& gather, float* policy_out, float* wdl_out,
                      const uint64_t* masks, const float* vals) {
  Impl& I=*p_;
  // ---- lc0bench 0012: the upload, on its own slot and its own stream, BEFORE the forward mutex ----
  int slot = 0;
  std::unique_lock<std::mutex> slk;
  const bool aup = I.asyncup && masks && N <= I.slotcap;   // slotcap never changes after the ctor
  if (aup) {
    { std::lock_guard<std::mutex> g(I.slotpick); slot = I.slotrr++ % Impl::NSLOT; }
    slk = std::unique_lock<std::mutex>(I.slotmtx[slot]);   // ALWAYS taken before I.mtx
    CK(cudaSetDevice(I.device));
    CK(cudaStreamWaitEvent(I.upst[slot], I.slotFree[slot], 0));   // the last user has finished reading it
    CK(cudaMemcpyAsync(I.sMask[slot],masks,(size_t)N*112*sizeof(uint64_t),cudaMemcpyHostToDevice,I.upst[slot]));
    CK(cudaMemcpyAsync(I.sStgV[slot],vals,(size_t)N*112*sizeof(float),cudaMemcpyHostToDevice,I.upst[slot]));
    copyTypeConverted(I.sVal[slot],I.sStgV[slot],N*112,I.upst[slot]);
    expandPlanes_NCHW<half_t>(I.sPlanes[slot],I.sMask[slot],I.sVal[slot],N*112,I.upst[slot]);
    k_flat12<<<N,64,0,I.upst[slot]>>>(I.sFlat[slot],I.sMask[slot],I.sVal[slot],N);
    CK(cudaEventRecord(I.upDone[slot], I.upst[slot]));
  }
  std::lock_guard<std::mutex> lk(I.mtx); CK(cudaSetDevice(I.device)); I.ensure(N); cublasHandle_t cub=I.cub;
  half_t *const PL = aup ? I.sPlanes[slot] : I.dPlanes;   // what the stem and the route read
  half_t *const FL = aup ? I.sFlat[slot] : I.dFlat;
  const int d=I.d,H=I.H,hd=I.hd,dff=I.dff,E=I.E,ed=I.ed,pd=I.pd,bank=I.bank; const float al=I.alpha;
  const size_t T=(size_t)N*64*d; const int R=N*64;

  // ---- optional phase profiling (HERO_PROFILE=1): stem/route/trunk/heads ----
  static const bool prof = getenv("HERO_PROFILE") != nullptr;
  static cudaEvent_t E0=0,Es=0,Er=0,Et=0,Eh=0,Eu=0;   // Eu: mfu60 upload split
  static double a_stem=0,a_route=0,a_trunk=0,a_heads=0,a_up=0; static long prof_calls=0;
  if (prof && !E0){ cudaEventCreate(&E0);cudaEventCreate(&Es);cudaEventCreate(&Er);cudaEventCreate(&Et);cudaEventCreate(&Eh);cudaEventCreate(&Eu); }
  // mfu60: HERO_PROFILE=2 -> per-operator trunk timing (docs/audit/herod4_engine_384.md)
  static const bool prof2 = prof && atoi(getenv("HERO_PROFILE")) >= 2;
  static std::vector<cudaEvent_t> evop; static double a_op[8]={0,0,0,0,0,0,0,0};
  if (prof2 && evop.empty()) { evop.resize((size_t)I.L*9); for (auto& e : evop) cudaEventCreate(&e); }
  auto opev = [&](int li, int k){ if (prof2) cudaEventRecord(evop[(size_t)li*9+k],I.rs); };
  if (prof) cudaEventRecord(E0,I.rs);

  if (aup) { // lc0bench 0012: already uploaded, on upst[slot]; just order the forward behind it
    CK(cudaStreamWaitEvent(I.rs, I.upDone[slot], 0));
  } else if (masks) { // lc0bench 0009: upload the PACKED planes (mask + value per plane,
    // 0.52 MB at N=384 against 12.2) and expand on device, as lc0's own cuda backend
    // does. expandPlanes_NCHW writes 0 on an unset bit = the host buffer's zero init.
    CK(cudaMemcpyAsync(I.dMask,masks,(size_t)N*112*sizeof(uint64_t),cudaMemcpyHostToDevice,I.rs));
    CK(cudaMemcpyAsync(I.stgV,vals,(size_t)N*112*sizeof(float),cudaMemcpyHostToDevice,I.rs));
    copyTypeConverted(I.dVal,I.stgV,N*112,I.rs);
    expandPlanes_NCHW<half_t>(I.dPlanes,I.dMask,I.dVal,N*112,I.rs);
    k_flat12<<<N,64,0,I.rs>>>(I.dFlat,I.dMask,I.dVal,N);
  } else { // upload planes (fp32 -> fp16) via copyTypeConverted. mfu60: the staging buffers
    // are allocated ONCE in ensure(); the old per-call cudaMalloc/cudaFree pair cost
    // ~10 ms of a 77 ms forward at mb 384 (cudaFree synchronizes the device).
    CK(cudaMemcpyAsync(I.stgP,planes_nchw,(size_t)N*112*64*sizeof(float),cudaMemcpyHostToDevice,I.rs));
    copyTypeConverted(I.dPlanes,I.stgP,(int)((size_t)N*112*64),I.rs);
    CK(cudaMemcpyAsync(I.stgF,flat12,(size_t)N*768*sizeof(float),cudaMemcpyHostToDevice,I.rs));
    copyTypeConverted(I.dFlat,I.stgF,(int)((size_t)N*768),I.rs);
  }
  if (prof) cudaEventRecord(Eu,I.rs);   // upload done (mfu60)

  // lc0bench 0010: uploaded once, BEFORE the capture decision (cudaMalloc is not capturable)
  if (!I.dGather) { CK(cudaMalloc(&I.dGather,1858*sizeof(int)));
    CK(cudaMemcpy(I.dGather,gather.data(),1858*sizeof(int),cudaMemcpyHostToDevice)); }

  // lc0bench 0010: the stem/route/trunk/heads as one callable, so it can be captured.
  auto forward_body = [&]() {
  // ---- stem ----
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,128,N,768,1.f,I.pp0,768,FL,768,0.f,I.pos128,128);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,8192,N,128,1.f,I.pp1,128,I.pos128,128,0.f,I.pos8192,8192);
  inputPreprocessForAttentionBody<half_t>(I.cat240,PL,I.pos8192,N,112,128,true,I.rs);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,240,1.f,I.emb,240,I.cat240,240,0.f,I.emb_d,d);
  LayerNorm<half_t>(R,d,I.e_out,I.emb_d,I.zbuf,(half_t*)nullptr,I.eln,I.zbuf,1e-3f,1.f,ACTIVATION_MISH,I.rs);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,ed,R,d,1.f,I.eu,d,I.e_out,d,0.f,I.up_h,ed);
  addBiasBatched<half_t>(I.up_h,I.up_h,I.zbuf,1,R,ed,ACTIVATION_MISH,I.rs);
  gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,ed,1.f,I.edn,ed,I.up_h,ed,0.f,I.dn_h,d);
  LayerNorm<half_t>(R,d,I.x,I.dn_h,I.zbuf,I.e_out,I.efln,I.zbuf,1e-3f,al,ACTIVATION_NONE,I.rs);

  if (prof) cudaEventRecord(Es,I.rs);   // stem done
  // ---- route -> expert-contiguous order[] + host offsets[] ----
  std::vector<int> off(E+1,0);
  if (E<=13) {   // 13-class: cheap host route (unchanged)
    std::vector<int> route; I.route(planes_nchw,N,route);
    std::vector<int> order(R), cnt(E,0);
    for (int r=0;r<R;r++) cnt[route[r]]++;
    for (int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
    { std::vector<int> cur(off.begin(),off.end()-1); for(int r=0;r<R;r++) order[cur[route[r]]++]=r; }
    CK(cudaMemcpyAsync(I.dOrder,order.data(),R*sizeof(int),cudaMemcpyHostToDevice,I.rs));
  } else {       // 28-class: fully device-resident route + counting sort
    CK(cudaMemsetAsync(I.dAtt_o,0,R,I.rs)); CK(cudaMemsetAsync(I.dAtt_t,0,R,I.rs));
    k_occ_piece<<<dim3(N,1),64,0,I.rs>>>(PL,I.dOcc,I.dPiece,N);
    k_attackers<<<dim3(N,1),64,0,I.rs>>>(PL,I.dOcc,I.dAtt_o,I.dAtt_t,N);
    k_route_id<<<dim3(N,1),64,0,I.rs>>>(I.dPiece,I.dAtt_o,I.dAtt_t,I.dRoute,N);
    CK(cudaMemsetAsync(I.dCnt,0,(E+1)*sizeof(int),I.rs));
    k_hist<<<(R+255)/256,256,0,I.rs>>>(I.dRoute,I.dCnt,R);
    if (I.gffn) {   // lc0bench 0006: offsets, the scatter cursor and the tile->class map,
      heroGFFNMap(I.dCnt, I.dOffs, I.dCur, I.dTmap, E, R/heroGFFNTileRows()+E, I.rs);
    } else {        // all on device. The D2H below BLOCKS: route's one sync with the host
    CK(cudaStreamSynchronize(I.rs));   // lc0bench 0010: the D2H below reads what the stream wrote
    std::vector<int> cnt(E); CK(cudaMemcpy(cnt.data(),I.dCnt,E*sizeof(int),cudaMemcpyDeviceToHost));
    for (int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
    CK(cudaMemcpyAsync(I.dCur,off.data(),E*sizeof(int),cudaMemcpyHostToDevice,I.rs));   // cur = start offsets
    }
    k_scatter_order<<<(R+255)/256,256,0,I.rs>>>(I.dRoute,I.dCur,I.dOrder,R);
  }
  if (prof) cudaEventRecord(Er,I.rs);   // route done

  // ---- trunk: 15 layers (fusedMHA + gather/scatter expert FFN) ----
  dim3 gd(R,((d>>3)+255)/256);   // int4-vectorized gather/scatter (8 halfs/thread)
  for (int li=0; li<I.L; li++){
    auto& t=I.lw[li]; opev(li,0);
    // q/k/v project d -> bank (=H*hd); weights are [bank,d]. bank may be >> d (bankdeep).
    if (!I.qkvb)   // lc0bench 0008
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,bank,R,d,1.f,t.qw,d,I.x,d,0.f,I.qd,bank);
    if (!I.qkvb)   // lc0bench 0008
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,bank,R,d,1.f,t.kw,d,I.x,d,0.f,I.kd,bank);
    if (!I.qkvb)   // lc0bench 0008
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,bank,R,d,1.f,t.vw,d,I.x,d,0.f,I.vd,bank);
    opev(li,1);   // q|k|v done
    if (I.qkvb) {   // lc0bench 0008: q|k|v and the band as one program; qd/kd/vd are gone
      if (I.qkvbw)   // lc0bench 0011: the warp-specialized entry, same contract and same output
        hero_qkv_band_ws(I.x, t.qw, t.kw, t.vw, t.bias, I.po, R, d, bank, H, I.rs);
      else
        hero_qkv_band(I.x, t.qw, t.kw, t.vw, t.bias, I.po, R, d, bank, H, I.rs);
    } else if (I.band) {
      heroBand(I.po, I.qd, I.kd, I.vd, t.bias, N, H, hd, I.rs);   // lc0bench 0005
    } else if (I.fused) {
      // broadcast the static (H,64,64) bias to all N (strideB=0) — no per-layer
      // N-broadcast write (was ~8GB/fwd at bs2048; the large-batch killer).
      fusedMHA<half_t>(I.po, I.qd, I.kd, I.vd, t.bias, N, H, hd, I.rs, true);
    } else {
      // non-fused path for hd not in {32,64,128} (e.g. run8d hd=5): batched-gemm
      // QK/AV + softmax(+bias). Validated layout (matches stem_gate / pre-fused).
      const float fac=1.f/sqrtf((float)hd), z0=0.f, o1=1.f;
      k_toheads<<<dim3(N,64,H),hd,0,I.rs>>>(I.qt,I.qd,H,hd);
      k_toheads<<<dim3(N,64,H),hd,0,I.rs>>>(I.kt,I.kd,H,hd);
      k_toheads<<<dim3(N,64,H),hd,0,I.rs>>>(I.vt,I.vd,H,hd);
      CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,hd,&fac,I.kt,CUDA_R_16F,hd,64*hd,I.qt,CUDA_R_16F,hd,64*hd,&z0,I.sca,CUDA_R_16F,64,64*64,N*H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
      k_softmax_bias<<<N*H*64,64,0,I.rs>>>(I.sca,t.bias,N*H*64,H*64);   // +static bias (broadcast) + softmax, no dBias
      CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,hd,64,64,&o1,I.vt,CUDA_R_16F,hd,64*hd,I.sca,CUDA_R_16F,64,64*64,&z0,I.ctx,CUDA_R_16F,hd,64*hd,N*H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
      k_fromheads<<<dim3(N,64,H),hd,0,I.rs>>>(I.po,I.ctx,H,hd);
    }
    opev(li,2);   // band (fusedMHA) done
    // out projects bank -> d; weight is [d,bank], po is bank-wide.
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,bank,1.f,t.ow,bank,I.po,bank,0.f,I.attn,d);
    opev(li,3);   // out projection done
    if (I.fastln) heroAddLN(I.xa,I.attn,I.x,t.l1g,R,d,1e-3f,al,I.rs);   // lc0bench 0007
    else
    LayerNorm<half_t>(R,d,I.xa,I.attn,I.zbuf,I.x,t.l1g,I.zbuf,1e-3f,al,ACTIVATION_NONE,I.rs);
    opev(li,4);   // add+LN1 done
    if (I.calib) k_chan_absmax<<<(d+255)/256,256,0,I.rs>>>(I.xa,I.calib_up+(size_t)li*d,R,d);   // up-input stats
    if (!I.gffn || I.no_agat)   // lc0bench 0006: the grouped up-GEMM gathers in its own A-load
    k_gather<<<gd,256,0,I.rs>>>(I.xs,I.xa,I.dOrder,d);
    opev(li,5);   // gather done
    if (I.int8) {   // int8 experts on NS streams — quant/dequant of one expert overlaps the
                    // int8 gemm of another (the v1 fix: v1 was serial, so overhead wasn't hidden)
      k_quant_rows<<<R,256,0,I.rs>>>(I.xs,I.xs_i8,I.xrow_inv,R,d,t.sq_up_inv);   // hoisted input quant
      cudaEventRecord(I.ev_gather, I.rs);
      for (int e=0;e<E;e++){ int m=off[e+1]-off[e]; if(!m) continue; size_t oo=off[e];
        cudaStream_t st=I.streams[e % Impl::NS];
        cudaStreamWaitEvent(st, I.ev_gather, 0);
        cublasSetStream(cub, st);
        gemm_i8(cub,dff,m,d, t.up_i8+(size_t)e*dff*d,d, I.xs_i8+oo*d,d, I.ffgh_i32+oo*dff,dff);
        k_dq_mish_q<<<m,256,dff*sizeof(float),st>>>(I.ffgh_i32+oo*dff,I.gh_i8+oo*dff,I.ghrow_inv+oo,t.up_inv+(size_t)e*dff,I.xrow_inv+oo,t.sq_dn_inv,dff,m);
        gemm_i8(cub,d,m,dff, t.dn_i8+(size_t)e*d*dff,dff, I.gh_i8+oo*dff,dff, I.ys_i32+oo*d,d);
        k_dequant<<<dim3((d+255)/256,m),256,0,st>>>(I.ys_i32+oo*d,I.ys+oo*d,t.dn_inv+(size_t)e*d,I.ghrow_inv+oo,d,m,0); }
      for (int s=0;s<Impl::NS;s++){ cudaEventRecord(I.ev_done[s], I.streams[s]); cudaStreamWaitEvent(I.rs, I.ev_done[s], 0); }
      cublasSetStream(cub, I.rs);
    } else {   // fp16 experts run CONCURRENTLY across NS streams (independent) — key for hero5 scaling
      // lc0bench 0010: with 0006 the grouped GEMMs launch on NO side stream, so the fork/join
      // below is 16 no-op event operations per layer -- and an event recorded on a stream that is
      // not part of the capture invalidates it. Both are now the per-expert loop's alone.
      if (!I.gffn) cudaEventRecord(I.ev_gather, I.rs);
      if (I.gffn) {   // lc0bench 0006: two grouped GEMMs, mish and the scatter fused
        const int nt_ = R/heroGFFNTileRows()+E;
        heroGFFNUp(I.ffgh, I.no_agat ? I.xs : I.xa, t.upp, I.dTmap, I.dOffs,
                   I.no_agat ? nullptr : I.dOrder, R, I.dp, d, nt_, I.rs);
        heroGFFNDown(I.ffn, I.ffgh, t.dn, I.dTmap, I.dOffs, I.dOrder, R, d, dff, I.dp, nt_, I.rs);
      } else
      for (int e=0;e<E;e++){ int m=off[e+1]-off[e]; if(!m) continue;
        cudaStream_t st=I.streams[e % Impl::NS];
        cudaStreamWaitEvent(st, I.ev_gather, 0);          // each expert waits for the gather
        if (I.cutlass_ffn) {   // fused mish epilogue in the up-gemm (no separate addBias-mish)
          cutlassFFNUpMish(I.xs+(size_t)off[e]*d, t.up+(size_t)e*dff*d, I.ffgh+(size_t)off[e]*dff, m, d, dff, st);
        } else {
          cublasSetStream(cub, st);
          gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,m,d,1.f,t.up+(size_t)e*dff*d,d,I.xs+(size_t)off[e]*d,d,0.f,I.ffgh+(size_t)off[e]*dff,dff);
          addBiasBatched<half_t>(I.ffgh+(size_t)off[e]*dff,I.ffgh+(size_t)off[e]*dff,I.zbuf,1,m,dff,ACTIVATION_MISH,st);
        }
        cublasSetStream(cub, st);
        gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,m,dff,1.f,t.dn+(size_t)e*d*dff,dff,I.ffgh+(size_t)off[e]*dff,dff,0.f,I.ys+(size_t)off[e]*d,d); }
      if (!I.gffn) for (int s=0;s<Impl::NS;s++){ cudaEventRecord(I.ev_done[s], I.streams[s]); cudaStreamWaitEvent(I.rs, I.ev_done[s], 0); }
      cublasSetStream(cub, I.rs);                           // back to Run's stream; scatter after all experts
      opev(li,6);   // expert GEMMs (8 streams, joined above) done
    }
    if (I.calib) k_chan_absmax<<<(dff+255)/256,256,0,I.rs>>>(I.ffgh,I.calib_dn+(size_t)li*dff,R,dff);   // down-input stats
    if (!I.gffn)   // lc0bench 0006: the grouped down-GEMM already stored through dOrder
    k_scatter<<<gd,256,0,I.rs>>>(I.ffn,I.ys,I.dOrder,d);
    opev(li,7);   // scatter done
    if (I.gffn && I.gffn_check && li==0) {   // run the loop path too and score the gap
      CK(cudaStreamSynchronize(I.rs));
      std::vector<int> o2(E+1); CK(cudaMemcpy(o2.data(),I.dOffs,(E+1)*sizeof(int),cudaMemcpyDeviceToHost));
      k_gather<<<gd,256,0,I.rs>>>(I.xs,I.xa,I.dOrder,d);   // the loop path needs the staged copy
      CK(cudaMemcpyAsync(I.ffnref,I.ffn,(size_t)R*d*sizeof(half_t),cudaMemcpyDeviceToDevice,I.rs));
      for (int e=0;e<E;e++){ int m=o2[e+1]-o2[e]; if(!m) continue;
        gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,m,d,1.f,t.up+(size_t)e*dff*d,d,I.xs+(size_t)o2[e]*d,d,0.f,I.ffgh+(size_t)o2[e]*dff,dff);
        addBiasBatched<half_t>(I.ffgh+(size_t)o2[e]*dff,I.ffgh+(size_t)o2[e]*dff,I.zbuf,1,m,dff,ACTIVATION_MISH,I.rs);
        gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,m,dff,1.f,t.dn+(size_t)e*d*dff,dff,I.ffgh+(size_t)o2[e]*dff,dff,0.f,I.ys+(size_t)o2[e]*d,d); }
      k_scatter<<<gd,256,0,I.rs>>>(I.ffn,I.ys,I.dOrder,d);
      float h2[2]={0,0}; heroGFFNCompare(I.ffnref,I.ffn,R*d,I.cmpd,h2);
      fprintf(stderr,"HEROGFFN layer0 grouped-vs-loop: max|d| %.5f  max rel %.5f  (n=%d)\n",h2[0],h2[1],R*d);
      CK(cudaMemcpyAsync(I.ffn,I.ffnref,(size_t)R*d*sizeof(half_t),cudaMemcpyDeviceToDevice,I.rs));
    }
    if (I.fastln) heroAddLN(I.x,I.ffn,I.xa,t.l2g,R,d,1e-3f,al,I.rs);    // lc0bench 0007
    else
    LayerNorm<half_t>(R,d,I.x,I.ffn,I.zbuf,I.xa,t.l2g,I.zbuf,1e-3f,al,ACTIVATION_NONE,I.rs);  // x = next input
    opev(li,8);   // add+LN2 done
  }
  if (prof) cudaEventRecord(Et,I.rs);   // trunk done

  // ---- heads (fully device-resident) ----
  const float sc = 1.f/sqrtf((float)pd), z=0.f, o=1.f;
  // policy: tp = mish(pol_embed(x)+b); qp,kp = pol_q/k(tp)+b; scp = qp.kp^T*sc; +promo; gather
  I.head_gemm(I.h_pe,d,pd,I.h_peb,true,I.x,R,I.tp);
  I.head_gemm(I.h_pq,pd,pd,I.h_pqb,false,I.tp,R,I.qp);
  I.head_gemm(I.h_pk,pd,pd,I.h_pkb,false,I.tp,R,I.kp);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,pd,&sc,I.kp,CUDA_R_16F,pd,64*pd,I.qp,CUDA_R_16F,pd,64*pd,&z,I.scp,CUDA_R_16F,64,64*64,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  { dim3 pg(N,8); k_promo<<<pg,4,0,I.rs>>>(I.promo,I.kp,I.h_ppo,N,pd); }
  { dim3 gg(N,(1858+255)/256); k_pol_gather<<<gg,256,0,I.rs>>>(I.d_pol,I.scp,I.promo,I.dGather,N); }
  // value: tv = mish(val_embed(x)); qv,kv,vv; softmax(qv.kv^T*sc) then .vv, mean over queries
  I.head_gemm(I.h_ve,d,pd,I.h_veb,true,I.x,R,I.tv);
  I.head_gemm(I.h_vq,pd,pd,I.h_vqb,false,I.tv,R,I.qv);
  I.head_gemm(I.h_vk,pd,pd,I.h_vkb,false,I.tv,R,I.kv);
  I.head_gemm(I.h_vv,pd,3,nullptr,false,I.tv,R,I.vvh);
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,pd,&sc,I.kv,CUDA_R_16F,pd,64*pd,I.qv,CUDA_R_16F,pd,64*pd,&z,I.scv,CUDA_R_16F,64,64*64,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  Softmax<half_t>(N*64,64,I.scv,I.scv,(half_t*)nullptr,I.rs);
  // vout(N,64,3) = scv(64x64) . vv(64x3), per position (see .cu notes for the layout)
  CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,3,64,64,&o,I.vvh,CUDA_R_16F,3,64*3,I.scv,CUDA_R_16F,64,64*64,&z,I.vout,CUDA_R_16F,3,64*3,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  k_wdl_mean<<<N,3,0,I.rs>>>(I.d_wdl,I.vout,N);
  if (prof) cudaEventRecord(Eh,I.rs);   // heads done
  };   // end forward_body (lc0bench 0010)

  // ---- lc0bench 0010: replay the graph for this N, or capture it on the 2nd sighting ----
  // Off under HERO_PROFILE (the per-operator events live inside the region), for the 13-class
  // host route and the per-expert loop (both take host values mid-forward), for int8/calib/
  // GFFN_CHECK, and under HERO_NO_GRAPH=1 (the same-binary A/B).
  const bool cap_ok = !I.nograph && !prof && I.gffn && !I.gffn_check && !I.int8 && !I.calib && E>13;
  cudaGraphExec_t ex = nullptr;
  const int gkey = N * Impl::NSLOT + slot;   // lc0bench 0012: a graph bakes in the slot it captured
  if (cap_ok) { auto it = I.graphs.find(gkey); if (it != I.graphs.end()) ex = it->second; }
  if (ex) {
    CK(cudaGraphLaunch(ex, I.rs));
  } else {
    // the FIRST call at this N runs eagerly: cuBLAS workspaces, the >48 KB SMEM opt-in and
    // lc0's own lazy allocations must all have happened before a capture begins.
    bool cap = cap_ok && ++I.seen[gkey] >= 2;
    if (cap && cudaStreamBeginCapture(I.rs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      cap = false; I.nograph = true;
      fprintf(stderr,"Hero 0010: cudaStreamBeginCapture failed at N=%d; eager from here\n", N);
    }
    forward_body();
    if (cap) {
      cudaGraph_t g = nullptr;
      if (cudaStreamEndCapture(I.rs,&g) == cudaSuccess && g &&
          cudaGraphInstantiateWithFlags(&ex,g,0) == cudaSuccess) {
        I.graphs[gkey] = ex;
        fprintf(stderr,"Hero 0010: captured a graph for N=%d\n", N);   // the A/B needs proof it ran
        CK(cudaGraphLaunch(ex, I.rs));          // the captured body did NOT run; replay it now
      } else {
        I.nograph = true; I.drop_graphs();     // one failure and the backend stays eager
        fprintf(stderr,"Hero 0010: graph capture failed at N=%d; eager from here\n", N);
        forward_body();
      }
      if (g) cudaGraphDestroy(g);
    }
  }
  if (aup) CK(cudaEventRecord(I.slotFree[slot], I.rs));   // lc0bench 0012: the slot is readable again
  CK(cudaStreamSynchronize(I.rs));
  if (prof) { float ms; long c=++prof_calls;
    cudaEventElapsedTime(&ms,E0,Eu); a_up+=ms;
    cudaEventElapsedTime(&ms,Eu,Es); a_stem+=ms;  cudaEventElapsedTime(&ms,Es,Er); a_route+=ms;
    cudaEventElapsedTime(&ms,Er,Et); a_trunk+=ms; cudaEventElapsedTime(&ms,Et,Eh); a_heads+=ms;
    if (prof2) { for (size_t i=0;i<evop.size();i+=9) for (int k=0;k<8;k++){
        float m; cudaEventElapsedTime(&m,evop[i+k],evop[i+k+1]); a_op[k]+=m; } }
    if (prof2 && c%50==0) fprintf(stderr,"HEROOP N=%d/%ld-avg ms: qkv %.3f | band %.3f | out %.3f | ln1 %.3f | gather %.3f | experts %.3f | scatter %.3f | ln2 %.3f (trunk sum %.2f)\n",
      N,c,a_op[0]/c,a_op[1]/c,a_op[2]/c,a_op[3]/c,a_op[4]/c,a_op[5]/c,a_op[6]/c,a_op[7]/c,
      (a_op[0]+a_op[1]+a_op[2]+a_op[3]+a_op[4]+a_op[5]+a_op[6]+a_op[7])/c);
    if (c%50==0) fprintf(stderr,"HEROPROF N=%d/50-avg: stem %.2f | route %.2f | trunk %.2f | heads %.2f ms (sum %.1f)\n",
      N, a_stem/c, a_route/c, a_trunk/c, a_heads/c, (a_up+a_stem+a_route+a_trunk+a_heads)/c);
    if (c%50==0) fprintf(stderr,"HEROUP N=%d/%ld-avg: upload %.2f ms\n", N, c, a_up/c); }
  CK(cudaMemcpy(policy_out,I.d_pol,(size_t)N*1858*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(wdl_out,I.d_wdl,(size_t)N*3*sizeof(float),cudaMemcpyDeviceToHost));
}

void HeroForward::DebugRoute(const float* planes_nchw, int N, std::vector<int>& out) {
  Impl& I=*p_; std::lock_guard<std::mutex> lk(I.mtx); CK(cudaSetDevice(I.device)); I.ensure(N); const int R=N*64;
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

void HeroForward::WriteCalib(const char* path) {
  Impl& I=*p_; if(!I.calib_up){ fprintf(stderr,"WriteCalib: not in calib mode\n"); return; }
  std::vector<float> up((size_t)I.L*I.d), dn((size_t)I.L*I.dff);
  CK(cudaMemcpy(up.data(),I.calib_up,up.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dn.data(),I.calib_dn,dn.size()*sizeof(float),cudaMemcpyDeviceToHost));
  FILE* f=fopen(path,"wb"); fwrite(up.data(),sizeof(float),up.size(),f); fwrite(dn.data(),sizeof(float),dn.size(),f); fclose(f);
  fprintf(stderr,"WriteCalib: wrote %s (L=%d d=%d dff=%d)\n",path,I.L,I.d,I.dff);
}

}  // namespace hero
}  // namespace lczero
