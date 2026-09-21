// The `-hero` network backend (see HERO_BACKEND.md). Loads .htw weights and runs
// the device-resident Hero forward (hero_forward.cu): stem -> 15 routed-expert
// layers (fusedMHA attention) -> policy + WDL. Registered so
// `--backend=hero --weights=x.htw` routes here.
#include <cmath>
#include <cstdlib>
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
  HeroNetwork(const std::string& path, int gpu)
      : weights_(hero::LoadHeroWeights(path)), forward_(weights_, gpu) {
    capabilities_.input_format =
        pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE;  // lc0bench patch 0001
    capabilities_.output_format = pblczero::NetworkFormat::OUTPUT_WDL;
    capabilities_.moves_left = pblczero::NetworkFormat::MOVES_LEFT_NONE;
    // invert lc0's 4288->1858 scatter map into the 1858-entry gather we need
    gather_.assign(1858, 0);
    for (int i = 0; i < static_cast<int>(std::size(kAttnPolicyMap)); ++i) {
      short j = kAttnPolicyMap[i];
      if (j >= 0) gather_[j] = i;
    }
    // lc0bench 0009: upload the packed planes and expand on device. The 13-class
    // route is host-side off the dense planes, so it keeps the old path.
    packed_ = weights_.classes > 13 && getenv("HERO_NO_PACKED") == nullptr;
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
  bool packed() const { return packed_; }   // lc0bench 0009
  const std::vector<int>& gather() const { return gather_; }

 private:
  hero::HeroWeights weights_;
  hero::HeroForward forward_;
  bool packed_ = false;   // lc0bench 0009
  std::vector<int> gather_;
  NetworkCapabilities capabilities_;
};

void HeroNetworkComputation::ComputeBlocking() {
  const int N = static_cast<int>(inputs_.size());
  if (N == 0) return;
  policy_.assign((size_t)N * 1858, 0.f);
  std::vector<float> wdl((size_t)N * 3, 0.f);
  // expand InputPlanes (mask+value) -> dense planes_nchw [N,112,64] and the
  // square-major first-12 flat [N,768] the positional preproc consumes.
  auto dense = [&](std::vector<float>& planes_nchw, std::vector<float>& flat12) {
    planes_nchw.assign((size_t)N * 112 * 64, 0.f);
    flat12.assign((size_t)N * 768, 0.f);
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
  };
  if (!network_->packed()) {
    std::vector<float> planes_nchw, flat12;
    dense(planes_nchw, flat12);
    network_->forward().Run(planes_nchw.data(), flat12.data(), N,
                            network_->gather(), policy_.data(), wdl.data());
  } else {
    // lc0bench 0009: hand Run() the packed planes (one 64-bit mask + one float each,
    // 0.52 MB at N=384) and let the device expand them. Planes past planes.size() are
    // packed as mask 0 / value 0 -- exactly what the dense path's zero init left there.
    std::vector<uint64_t> masks((size_t)N * 112, 0ull);
    std::vector<float> vals((size_t)N * 112, 0.f);
    for (int n = 0; n < N; ++n) {
      const auto& planes = inputs_[n];
      for (int c = 0; c < 112 && c < (int)planes.size(); ++c) {
        masks[(size_t)n * 112 + c] = planes[c].mask;
        vals[(size_t)n * 112 + c] = planes[c].value;
      }
    }
    // GATE 0009 (HERO_PACKED_CHECK=1): the same batch through BOTH paths in the same
    // process, compared bitwise on the policy logits and the WDL the search consumes.
    static const bool chk = getenv("HERO_PACKED_CHECK") != nullptr;
    std::vector<float> pref, wref;
    if (chk) {
      std::vector<float> pl, fl;
      dense(pl, fl);
      pref.assign((size_t)N * 1858, 0.f);
      wref.assign((size_t)N * 3, 0.f);
      network_->forward().Run(pl.data(), fl.data(), N, network_->gather(), pref.data(), wref.data());
    }
    network_->forward().Run(nullptr, nullptr, N, network_->gather(), policy_.data(),
                            wdl.data(), masks.data(), vals.data());
    if (chk) {
      double mp = 0, mw = 0;
      size_t nd = 0;
      for (size_t i = 0; i < pref.size(); ++i) {
        double e = std::fabs((double)pref[i] - policy_[i]);
        if (e > mp) mp = e;
        if (pref[i] != policy_[i]) ++nd;
      }
      for (size_t i = 0; i < wref.size(); ++i) {
        double e = std::fabs((double)wref[i] - wdl[i]);
        if (e > mw) mw = e;
      }
      CERR << "HEROPACK N=" << N << " dense-vs-packed: policy max|d| " << mp
           << " non-bitwise " << nd << " of " << pref.size() << " | wdl max|d| " << mw;
    }
  }
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
  return std::make_unique<HeroNetwork>(path, options.GetOrDefault<int>("gpu", 0));
}

REGISTER_NETWORK("hero", MakeHeroNetwork, 50)

}  // namespace
}  // namespace lczero
