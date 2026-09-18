// Forward gate for SmoothQuant. Modes:
//   <htw> <planes.npy> <out.bin>   run HeroForward::Run (fp16/int8/SQ by env), write
//                                  [int N][policy N*1858][wdl N*3]; if HERO_CALIB also
//                                  writes <out.bin>.calib for SmoothQuant.
//   cmp <a.bin> <b.bin>            compare two runs: top-1 agreement + wdl diff.
// planes.npy is lc0 NCHW image (N,112,8,8) == planes_nchw [N,112,64] flat. CudaError stubbed.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
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
static int am(const float* p){ int b=0; for(int m=1;m<1858;m++) if(p[m]>p[b]) b=m; return b; }

int main(int argc,char** argv){
  if(argc>=4 && !strcmp(argv[1],"cmp")){
    std::ifstream A(argv[2],std::ios::binary), B(argv[3],std::ios::binary);
    int na,nb; A.read((char*)&na,4); B.read((char*)&nb,4); if(na!=nb){printf("N mismatch\n");return 1;}
    std::vector<float> pa((size_t)na*1858),pb((size_t)na*1858),wa(na*3),wb(na*3);
    A.read((char*)pa.data(),pa.size()*4); B.read((char*)pb.data(),pb.size()*4);
    A.read((char*)wa.data(),wa.size()*4); B.read((char*)wb.data(),wb.size()*4);
    int top1=0; double vw=0,rmw=0;
    for(int n=0;n<na;n++){ if(am(&pa[(size_t)n*1858])==am(&pb[(size_t)n*1858])) top1++;
      for(int c=0;c<3;c++){ vw=fmax(vw,fabs((double)wa[n*3+c]-wb[n*3+c])); rmw=fmax(rmw,fabs((double)wa[n*3+c])); } }
    printf("CMP %s vs %s: top1-agree=%d/%d  wdl worst|d|=%.4f (rel %.3f)\n", argv[2],argv[3],top1,na,vw,rmw>0?vw/rmw:0);
    printf("%s\n", (top1>=na*0.95 && (rmw>0?vw/rmw:0)<0.08)?"CMP_OK":"CMP_DEGRADED");
    return 0;
  }
  if(argc<4){ printf("usage: hero_fwd_gate <htw> <planes.npy> <out.bin>  |  cmp <a.bin> <b.bin>\n"); return 2; }
  std::vector<int> ps; auto nchw=load_npy(argv[2],ps);   // (N,112,8,8) == [N,112,64] flat
  int N=ps[0];
  std::vector<float> flat12((size_t)N*768,0.f);
  for(int n=0;n<N;n++)for(int c=0;c<12;c++)for(int s=0;s<64;s++) flat12[(size_t)n*768+s*12+c]=nchw[((size_t)n*112+c)*64+s];
  // gather map is identity-load from the oracle? no — reuse lc0's; here we only need
  // top-1 self-consistency, and the gather is fixed, so a placeholder identity works
  // for RELATIVE comparison IF both runs use the same map. Use 0..1857 (stable).
  std::vector<int> gmap(1858); for(int i=0;i<1858;i++) gmap[i]=i;
  lczero::hero::HeroWeights w=lczero::hero::LoadHeroWeights(argv[1]);
  fprintf(stderr,"net classes=%d N=%d int8=%s sq=%s calib=%s\n",w.classes,N,
    getenv("HERO_INT8")?"1":"0",getenv("HERO_SQ")?"1":"0",getenv("HERO_CALIB")?"1":"0");
  lczero::hero::HeroForward fwd(w);
  std::vector<float> pol((size_t)N*1858),wdl((size_t)N*3);
  fwd.Run(nchw.data(),flat12.data(),N,gmap,pol.data(),wdl.data());
  std::ofstream out(argv[3],std::ios::binary);
  out.write((char*)&N,4); out.write((char*)pol.data(),pol.size()*4); out.write((char*)wdl.data(),wdl.size()*4);
  if(getenv("HERO_CALIB")){ std::string cf=std::string(argv[3])+".calib"; fwd.WriteCalib(cf.c_str()); }
  fprintf(stderr,"wrote %s\n",argv[3]);
  return 0;
}
