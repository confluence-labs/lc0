// Standalone gate: validate the DEVICE 28-class route (hero_forward.cu kernels)
// bit-exact vs the oracle route.npy. Critical because the graft makes most route
// errors invisible in forward output (enemy-bit / control differences share a
// parent expert) — so we must check route ids directly. Built with cutlass (the
// full hero_forward.cu compiles), CudaError stubbed (no layers.cc here).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
  if(argc<3){ printf("usage: hero_route_gate <28class.htw> <oracle28_dir>\n"); return 2; }
  std::vector<int> ps,rs;
  auto planes = load_npy(std::string(argv[2])+"/planes.npy", ps);   // (N,64,112) NHWC
  auto route  = load_npy(std::string(argv[2])+"/route.npy", rs);    // (N,64)
  int N=ps[0];
  // NHWC (N,64,112) -> NCHW (N,112,64)
  std::vector<float> nchw((size_t)N*112*64);
  for(int n=0;n<N;n++)for(int s=0;s<64;s++)for(int c=0;c<112;c++) nchw[((size_t)n*112+c)*64+s]=planes[((size_t)n*64+s)*112+c];
  lczero::hero::HeroWeights w = lczero::hero::LoadHeroWeights(argv[1]);
  printf("net classes=%d N=%d\n", w.classes, N);
  lczero::hero::HeroForward fwd(w);
  std::vector<int> dev; fwd.DebugRoute(nchw.data(), N, dev);
  int mism=0, first=-1;
  for(size_t i=0;i<dev.size();i++){ if(dev[i]!=(int)route[i]){ mism++; if(first<0){first=i; printf("  first mism idx=%zu dev=%d ref=%d\n",i,dev[i],(int)route[i]);} } }
  printf("DEVICE ROUTE28 %s: %d/%zu mismatches\n", mism==0?"PASS":"FAIL", mism, dev.size());
  return mism?1:0;
}
