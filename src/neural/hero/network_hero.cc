// The `-hero` network backend (see HERO_BACKEND.md). Loads .htw weights and runs
// the device-resident Hero forward (hero_forward.cu): stem -> 15 routed-expert
// layers (fusedMHA attention) -> policy + WDL. Registered so
// `--backend=hero --weights=x.htw` routes here.
#include <cmath>
#include <optional>
#include <vector>

#include "neural/factory.h"
#include "neural/hero/hero_forward.h"
#include "neural/hero/hero_weights.h"
#include "neural/network.h"
#include "neural/shared_params.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace {

class HeroNetwork;

class HeroNetworkComputation : public NetworkComputation {
 public:
  explicit HeroNetworkComputation(HeroNetwork* network) : network_(network) {}
  void AddInput(InputPlanes&& input) override { inputs_.push_back(std::move(input)); }
  int GetBatchSize() const override { return static_cast<int>(inputs_.size()); }
  void ComputeBlocking() override;  // defined after HeroNetwork
  float GetQVal(int sample) const override { return q_[sample]; }
  float GetDVal(int sample) const override { return d_[sample]; }
  float GetPVal(int sample, int move_id) const override {
    return policy_[static_cast<size_t>(sample) * 1858 + move_id];
  }
  float GetMVal(int) const override { return 0.0f; }

 private:
  HeroNetwork* network_;
  std::vector<InputPlanes> inputs_;
  std::vector<float> policy_;  // [N*1858] logits
  std::vector<float> q_, d_;   // [N]
};

class HeroNetwork : public Network {
 public:
  HeroNetwork(const std::string& path, const OptionsDict& /*options*/)
      : weights_(hero::LoadHeroWeights(path)), forward_(weights_) {
    capabilities_.input_format =
        pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2;
    capabilities_.output_format = pblczero::NetworkFormat::OUTPUT_WDL;
    capabilities_.moves_left = pblczero::NetworkFormat::MOVES_LEFT_NONE;
    // invert lc0's 4288->1858 scatter map into the 1858-entry gather we need
    gather_.assign(1858, 0);
    for (int i = 0; i < static_cast<int>(std::size(kAttnPolicyMap)); ++i) {
      short j = kAttnPolicyMap[i];
      if (j >= 0) gather_[j] = i;
    }
    CERR << "Hero net loaded: d=" << weights_.d << " layers=" << weights_.layers
         << " heads=" << weights_.heads << " dff=" << weights_.dff
         << " classes=" << weights_.classes;
  }
  const NetworkCapabilities& GetCapabilities() const override { return capabilities_; }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<HeroNetworkComputation>(this);
  }
  int GetMiniBatchSize() const override { return 384; }

  hero::HeroForward& forward() { return forward_; }
  const std::vector<int>& gather() const { return gather_; }

 private:
  hero::HeroWeights weights_;
  hero::HeroForward forward_;
  std::vector<int> gather_;
  NetworkCapabilities capabilities_;
};

void HeroNetworkComputation::ComputeBlocking() {
  const int N = static_cast<int>(inputs_.size());
  if (N == 0) return;
  // expand InputPlanes (mask+value) -> dense planes_nchw [N,112,64] and the
  // square-major first-12 flat [N,768] the positional preproc consumes.
  std::vector<float> planes_nchw((size_t)N * 112 * 64, 0.f);
  std::vector<float> flat12((size_t)N * 768, 0.f);
  for (int n = 0; n < N; ++n) {
    const auto& planes = inputs_[n];
    for (int c = 0; c < 112 && c < (int)planes.size(); ++c) {
      uint64_t mask = planes[c].mask;
      float val = planes[c].value;
      while (mask) {
        int s = __builtin_ctzll(mask);
        mask &= mask - 1;
        planes_nchw[((size_t)n * 112 + c) * 64 + s] = val;
        if (c < 12) flat12[(size_t)n * 768 + s * 12 + c] = val;
      }
    }
  }
  policy_.assign((size_t)N * 1858, 0.f);
  std::vector<float> wdl((size_t)N * 3, 0.f);
  network_->forward().Run(planes_nchw.data(), flat12.data(), N,
                          network_->gather(), policy_.data(), wdl.data());
  // WDL logits -> q (win-loss) and d (draw), softmax over the 3 outputs
  q_.assign(N, 0.f);
  d_.assign(N, 0.f);
  for (int n = 0; n < N; ++n) {
    float w = wdl[n * 3 + 0], dr = wdl[n * 3 + 1], l = wdl[n * 3 + 2];
    float mx = std::max(w, std::max(dr, l));
    float ew = std::exp(w - mx), ed = std::exp(dr - mx), el = std::exp(l - mx);
    float z = ew + ed + el;
    q_[n] = (ew - el) / z;
    d_[n] = ed / z;
  }
}

std::unique_ptr<Network> MakeHeroNetwork(const std::optional<WeightsFile>& /*w*/,
                                         const OptionsDict& options) {
  const std::string path =
      options.Get<std::string>(SharedBackendParams::kWeightsId);
  if (path.empty() || !hero::IsHeroWeightsFile(path)) {
    throw Exception("The hero backend requires a .htw weights file.");
  }
  return std::make_unique<HeroNetwork>(path, options);
}

REGISTER_NETWORK("hero", MakeHeroNetwork, 50)

}  // namespace
}  // namespace lczero
