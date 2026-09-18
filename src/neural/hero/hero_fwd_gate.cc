// Forward gate: run the REAL HeroForward::Run on the oracle planes and compare
// policy top-1 + WDL to the oracle. Used to validate int8 (HERO_INT8=1) accuracy
// vs fp16 — int8 changes the compute, so output differs by quantization error;
// we require top-1 move match to hold. CudaError stubbed (standalone, no layers.cc).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include "neural/hero/hero_forward.h"
#include "neural/hero/hero_weights.h"

namespace lczero { namespace cudnn_backend {
void CudaError(cudaError_t s,const char* f,const int& l){ if(s){fprintf(stderr,"CUDA %s (%s:%d)\n",cudaGetErrorString(s),f,l);exit(1);} }
}}

static std::vector<float> load_npy(const std::string& p, std::vector<int>& shape){
  std::ifstream f(p,std::ios::binary); char m[6]; f.read(m,6); uint8_t a,b; f.read((char*)&a,1); f.read((char*)&b,1);
  uint16_t hl; f.read((char*)&hl,2); std::string h(hl,0); f.read(&h[0],hl);
  size_t s=h.find("(")+1,e=h.find(")"); std::string sh=h.substr(s,e-s); shape.clear(); size_t p2=0;
  while(p2<sh.size()){ while(p2<sh.size()&&(sh[p2]==' '||sh[p2]==','))p2++; if(p2>=sh.size())break; shape.push_back(atoi(sh.c_str()+p2)); while(p2<sh.size()&&sh[p2]!=',')p2++; }
  std::vector<char> r((std::istreambuf_iterator<char>(f)),{});
  int w=h.find("u1")!=std::string::npos?1:(h.find("i8")!=std::string::npos||h.find("f8")!=std::string::npos)?8:4;
  bool ii=h.find("<i")!=std::string::npos||h.find("u1")!=std::string::npos; size_t n=r.size()/w; std::vector<float> o(n);
  for(size_t i=0;i<n;i++){ const char*q=&r[i*w]; if(w==1)o[i]=(float)(uint8_t)q[0]; else if(w==8){int64_t v;memcpy(&v,q,8);o[i]=v;} else if(ii){int32_t v;memcpy(&v,q,4);o[i]=v;} else {float v;memcpy(&v,q,4);o[i]=v;} }
  return o;
}

int main(int argc,char** argv){
  if(argc<3){ printf("usage: hero_fwd_gate <28class.htw> <oracle28_dir>\n"); return 2; }
  std::string od=argv[2]; std::vector<int> ps,gs;
  auto planes=load_npy(od+"/planes.npy",ps);           // (N,64,112) NHWC uint8
  auto gather=load_npy(od+"/gather.npy",gs);           // (1858,) int
  std::vector<int> dummy;
  auto refP=load_npy(od+"/policy.npy",dummy);          // (N,1858)
  auto refW=load_npy(od+"/wdl.npy",dummy);             // (N,3)
  int N=ps[0];
  std::vector<float> nchw((size_t)N*112*64,0.f), flat12((size_t)N*768,0.f);
  for(int n=0;n<N;n++)for(int s=0;s<64;s++)for(int c=0;c<112;c++){ float v=planes[((size_t)n*64+s)*112+c];
    nchw[((size_t)n*112+c)*64+s]=v; if(c<12) flat12[(size_t)n*768+s*12+c]=v; }
  std::vector<int> gmap(1858); for(int i=0;i<1858;i++) gmap[i]=(int)gather[i];

  lczero::hero::HeroWeights w=lczero::hero::LoadHeroWeights(argv[1]);
  printf("net classes=%d N=%d  (HERO_INT8=%s)\n", w.classes, N, getenv("HERO_INT8")?"1":"0");
  lczero::hero::HeroForward fwd(w);
  std::vector<float> pol((size_t)N*1858), wdl((size_t)N*3);
  fwd.Run(nchw.data(),flat12.data(),N,gmap,pol.data(),wdl.data());

  int top1=0; double vw=0,rmw=0;
  for(int n=0;n<N;n++){
    auto am=[&](const std::vector<float>& v){ int b=0; for(int m=1;m<1858;m++) if(v[(size_t)n*1858+m]>v[(size_t)n*1858+b]) b=m; return b; };
    if(am(pol)==am(refP)) top1++;
    for(int c=0;c<3;c++){ vw=fmax(vw,fabs((double)wdl[n*3+c]-refW[n*3+c])); rmw=fmax(rmw,fabs((double)refW[n*3+c])); }
  }
  printf("FWD gate: top1-match=%d/%d  wdl worst|d|=%.4f (of %.2f, rel %.3f)\n", top1,N, vw, rmw, vw/rmw);
  printf("%s\n", (top1==N && vw/rmw<0.05) ? "FWD_PASS" : "FWD_CHECK");
  return 0;
}
