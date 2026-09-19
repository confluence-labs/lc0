// INC4: device-resident Hero TRUNK forward (stem + 15 layers), timed on A100
// to measure nps (the payoff: beat BT4's 13.8k). Reuses the validated math
// (hero_stem_gate.cc) but replaces every host round-trip with a device kernel
// (gather/scatter/transpose). Expert routing is deterministic -> pre-sorted on
// the host once, offsets uploaded. FFN = per-expert cuBLAS loop (Ampere-safe;
// CUTLASS 2.x grouped is the next lever). Random weights (nps is shape-bound).
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "neural/hero/hero_weights.h"
#include "neural/backends/cuda/kernels.h"
using namespace lczero; using namespace lczero::cudnn_backend; using half_t=half;
namespace lczero { namespace cudnn_backend {
void CudaError(cudaError_t s,const char* f,const int& l){ if(s){fprintf(stderr,"CUDA %s (%s:%d)\n",cudaGetErrorString(s),f,l);exit(1);} }
}}
#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=(x);if(s){printf("CUBLAS %d: %d\n",__LINE__,(int)s);exit(1);}}while(0)

static void gemm(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,float a,const half_t* A,int lda,const half_t* B,int ldb,float b,half_t* C,int ldc){
  CB(cublasGemmEx(h,ta,tb,m,n,k,&a,A,CUDA_R_16F,lda,B,CUDA_R_16F,ldb,&b,C,CUDA_R_16F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)); }
static half_t* devrand(size_t n){ half_t* d; CK(cudaMalloc(&d,n*sizeof(half_t)));
  std::vector<half_t> h(n); for(size_t i=0;i<n;i++) h[i]=__float2half(((float)(i*2654435761u%1000)/1000.f-0.5f)*0.1f);
  CK(cudaMemcpy(d,h.data(),n*sizeof(half_t),cudaMemcpyHostToDevice)); return d; }

__global__ void k_gather(half_t* o,const half_t* in,const int* idx,int d){ int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)r*d+c]=in[(size_t)idx[r]*d+c]; }
__global__ void k_scatter(half_t* o,const half_t* in,const int* idx,int d){ int r=blockIdx.x,c=blockIdx.y*blockDim.x+threadIdx.x; if(c<d) o[(size_t)idx[r]*d+c]=in[(size_t)r*d+c]; }
__global__ void k_toheads(half_t* o,const half_t* in,int H,int hd){ int n=blockIdx.x,s=blockIdx.y,hh=blockIdx.z,e=threadIdx.x; if(e<hd) o[(((size_t)n*H+hh)*64+s)*hd+e]=in[((size_t)n*64+s)*(H*hd)+hh*hd+e]; }
__global__ void k_fromheads(half_t* o,const half_t* in,int H,int hd){ int n=blockIdx.x,s=blockIdx.y,hh=blockIdx.z,e=threadIdx.x; if(e<hd) o[((size_t)n*64+s)*(H*hd)+hh*hd+e]=in[(((size_t)n*H+hh)*64+s)*hd+e]; }

int main(int argc,char** argv){
  int B = argc>1?atoi(argv[1]):256;         // boards (leaf batch)
  int hd=32;
  int d = argc>2?atoi(argv[2]):1024;        // model width (hero3=1024, hero4=1280)
  int L = argc>3?atoi(argv[3]):15;          // layers (depth)
  int E = argc>4?atoi(argv[4]):13;          // experts
  int dff = argc>5?atoi(argv[5]):(d*5)/4;   // FFN width (hero3=1280, hero4=2048); default 1.25d
  int skew = argc>6?atoi(argv[6]):0;        // 0=uniform route; 1=REAL 13-class dist (54.5% empty)
  int H=d/hd, ed=d/2;                       // heads=d/32, embed_dff=0.5d
  double actM = L*(4.0*d*d + 2.0*(double)d*dff)/1e6;                    // active: attn 4d^2 + 1 expert 2*d*dff
  double totM = actM + L*(double)(E-1)*2.0*d*dff/1e6;                   // + the E-1 inactive experts
  char name[64]; cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0); snprintf(name,64,"%s",pr.name);
  double peak = strstr(name,"A100")?312e12: strstr(name,"H100")?989e12: strstr(name,"L4")?121e12: strstr(name,"L40")?181e12:100e12;
  cublasHandle_t cub; CB(cublasCreate(&cub)); CB(cublasSetMathMode(cub,CUBLAS_TENSOR_OP_MATH));
  int R=B*64; size_t T=(size_t)R*d;

  // deterministic route + host sort -> order[R], offsets[E+1] (uploaded once)
  std::vector<int> route(R), order(R), off(E+1,0), cnt(E,0);
  if(skew){ // REAL measured 13-class shares: empties 54.5%, pawns 11.1%x2, pieces 1.4-3.0%
    double frac[13]={0.545,0.111,0.028,0.029,0.030,0.014,0.016,0.111,0.028,0.029,0.030,0.014,0.016};
    int r=0; for(int e=0;e<E&&e<13;e++){ int ce=(int)(frac[e]*R); for(int k=0;k<ce&&r<R;k++) route[r++]=e; }
    while(r<R) route[r++]=0;                       // remainder -> empty class
    for(int r2=0;r2<R;r2++) cnt[route[r2]]++;
  } else {
    for(int r=0;r<R;r++){ route[r]=(r*1103515245u>>8)%E; cnt[route[r]]++; }
  }
  for(int e=0;e<E;e++) off[e+1]=off[e]+cnt[e];
  std::vector<int> cur(off.begin(),off.end()-1);
  for(int r=0;r<R;r++) order[cur[route[r]]++]=r;
  int *dOrder; CK(cudaMalloc(&dOrder,R*sizeof(int))); CK(cudaMemcpy(dOrder,order.data(),R*sizeof(int),cudaMemcpyHostToDevice));

  // weights (random, correct shapes) — persistent for all layers (reuse; nps is shape-bound)
  half_t *qw=devrand((size_t)d*d),*kw=devrand((size_t)d*d),*vw=devrand((size_t)d*d),*ow=devrand((size_t)d*d);
  half_t *l1=devrand(d),*l2=devrand(d),*zb=devrand(dff>d?dff:d);
  half_t *upw=devrand((size_t)E*dff*d),*dnw=devrand((size_t)E*d*dff),*bias=devrand((size_t)B*H*64*64);
  half_t *embU=devrand((size_t)ed*d),*embD=devrand((size_t)d*ed),*eg=devrand(d);
  for(int i=0;i<(dff>d?dff:d);i++){} // zb already random; set to 0 for bias
  { std::vector<half_t> z(dff>d?dff:d,__float2half(0.f)); CK(cudaMemcpy(zb,z.data(),z.size()*sizeof(half_t),cudaMemcpyHostToDevice)); }

  // buffers
  half_t *x,*qd,*kd,*vd,*qt,*kt,*vt,*sc,*ctx,*po,*attn,*xa,*xs,*gh,*ys,*ffn,*xn;
  auto A=[&](half_t** p,size_t n){ CK(cudaMalloc(p,n*sizeof(half_t))); };
  A(&x,T);A(&qd,T);A(&kd,T);A(&vd,T);A(&qt,T);A(&kt,T);A(&vt,T);A(&sc,(size_t)B*H*64*64);A(&ctx,T);A(&po,T);A(&attn,T);
  A(&xa,T);A(&xs,T);A(&gh,(size_t)R*dff);A(&ys,T);A(&ffn,T);A(&xn,T);
  x=devrand(T);  // pretend stem output
  float al=powf(2.f*L,-0.25f), fac=1.f/sqrtf((float)hd);
  dim3 gh_(R,(d+255)/256), gh2(R,(dff+255)/256), th(B,64,H);

  cudaEvent_t ea,eb,ec; cudaEventCreate(&ea);cudaEventCreate(&eb);cudaEventCreate(&ec);
  float t_attn=0,t_ffn=0;
  auto layer=[&](half_t* xin,half_t* xout){
    cudaEventRecord(ea);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,qw,d,xin,d,0.f,qd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,kw,d,xin,d,0.f,kd,d);
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,vw,d,xin,d,0.f,vd,d);
#ifdef USE_CUTLASS
    // fused flash attention: q/k/v (N,64,d) interleaved in, +bias(B,H,64,64),
    // scale+softmax+av fused, out (N,64,d) — no transposes, no batched gemms.
    fusedMHA<half_t>(po, qd, kd, vd, bias, B, H, hd, 0);
#else
    k_toheads<<<th,hd>>>(qt,qd,H,hd); k_toheads<<<th,hd>>>(kt,kd,H,hd); k_toheads<<<th,hd>>>(vt,vd,H,hd);
    float z=0,o=1;
    CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,64,64,hd,&fac,kt,CUDA_R_16F,hd,64*hd,qt,CUDA_R_16F,hd,64*hd,&z,sc,CUDA_R_16F,64,64*64,B*H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    Softmax<half_t>(B*H*64,64,sc,sc,bias,0);
    CB(cublasGemmStridedBatchedEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,hd,64,64,&o,vt,CUDA_R_16F,hd,64*hd,sc,CUDA_R_16F,64,64*64,&z,ctx,CUDA_R_16F,hd,64*hd,B*H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    k_fromheads<<<th,hd>>>(po,ctx,H,hd);
#endif
    gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,R,d,1.f,ow,d,po,d,0.f,attn,d);
    LayerNorm<half_t>(R,d,xa,attn,zb,xin,l1,zb,1e-3f,al,ACTIVATION_NONE,0);
    cudaEventRecord(eb);
    // FFN: gather by expert -> per-expert gemm up/mish/down -> scatter
    k_gather<<<gh_,256>>>(xs,xa,dOrder,d);
    for(int e=0;e<E;e++){ int m=off[e+1]-off[e]; if(!m) continue;
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,dff,m,d,1.f,upw+(size_t)e*dff*d,d,xs+(size_t)off[e]*d,d,0.f,gh+(size_t)off[e]*dff,dff);
      addBiasBatched<half_t>(gh+(size_t)off[e]*dff,gh+(size_t)off[e]*dff,zb,1,m,dff,ACTIVATION_MISH,0);
      gemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,d,m,dff,1.f,dnw+(size_t)e*d*dff,dff,gh+(size_t)off[e]*dff,dff,0.f,ys+(size_t)off[e]*d,d); }
    k_scatter<<<gh_,256>>>(ffn,ys,dOrder,d);
    LayerNorm<half_t>(R,d,xout,ffn,zb,xa,l2,zb,1e-3f,al,ACTIVATION_NONE,0);
    cudaEventRecord(ec); cudaEventSynchronize(ec);
    float a1,a2; cudaEventElapsedTime(&a1,ea,eb); cudaEventElapsedTime(&a2,eb,ec); t_attn+=a1; t_ffn+=a2;
  };
  auto fwd=[&](){ half_t* a=x; half_t* b=xn; for(int i=0;i<L;i++){ layer(a,b); half_t* t=a;a=b;b=t; } };

  for(int i=0;i<5;i++) fwd();          // warmup
  CK(cudaDeviceSynchronize());
  cudaEvent_t s0,s1; cudaEventCreate(&s0);cudaEventCreate(&s1);
  int IT=30; cudaEventRecord(s0); for(int i=0;i<IT;i++) fwd(); cudaEventRecord(s1); cudaEventSynchronize(s1);
  float ms; cudaEventElapsedTime(&ms,s0,s1); double per=ms/IT/1000.0;
  // trunk per-pos GFLOP: L*(qkvo 512 d^2 + FFN 256 d*dff), one active expert/token
  double gflop=(512.0*d*d + 256.0*(double)d*dff)*L/1e9;
  double posps=B/per, mfu=gflop*1e9*posps/peak;
  double tot=t_attn+t_ffn;
  printf("HEROSHAPE d=%d dff=%d H=%d L=%d E=%d act=%.0fM tot=%.0fM | %.0f pos/s  MFU=%.3f  ms=%.1f  [attn %.0f%% ffn %.0f%%]\n",
         d,dff,H,L,E,actM,totM,posps,mfu,per*1000, 100*t_attn/tot, 100*t_ffn/tot);
  printf("BENCH_DONE\n"); return 0;
}
