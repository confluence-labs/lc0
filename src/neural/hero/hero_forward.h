// Device-resident Hero forward (see hero_forward.cu). Constructed once from
// HeroWeights (uploads all weights to GPU); Run() does one batched forward.
#pragma once

#include <vector>

namespace lczero {
namespace hero {

struct HeroWeights;

class HeroForward {
 public:
  explicit HeroForward(const HeroWeights& w);
  ~HeroForward();
  HeroForward(const HeroForward&) = delete;
  HeroForward& operator=(const HeroForward&) = delete;

  // planes_nchw: [N,112,64] fp32 (dense planes, NCHW). flat12: [N,768] fp32
  // (square-major first-12 planes for the positional preproc). gather: the fixed
  // 1858-entry attention-policy move map (cat4288 -> 1858). Writes policy_out
  // [N*1858] and wdl_out [N*3] (logits; caller applies softmax as lc0 expects).
  void Run(const float* planes_nchw, const float* flat12, int N,
           const std::vector<int>& gather, float* policy_out, float* wdl_out);

 private:
  struct Impl;
  Impl* p_;
};

}  // namespace hero
}  // namespace lczero
