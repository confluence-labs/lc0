// The `-hero` network backend (see HERO_BACKEND.md). Increment 2: loads the
// .htw weights and registers, so `--backend=hero --weights=x.htw` routes here
// and parses the net. The forward (ComputeBlocking) is stubbed until increment 3.
#include <optional>
#include <vector>

#include "neural/factory.h"
#include "neural/hero/hero_weights.h"
#include "neural/network.h"
#include "neural/shared_params.h"
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
  void ComputeBlocking() override {
    throw Exception("hero backend: forward not implemented yet (increment 3)");
  }
  float GetQVal(int) const override { return 0.0f; }
  float GetDVal(int) const override { return 0.0f; }
  float GetPVal(int, int) const override { return 0.0f; }
  float GetMVal(int) const override { return 0.0f; }

 private:
  HeroNetwork* network_;
  std::vector<InputPlanes> inputs_;
};

class HeroNetwork : public Network {
 public:
  HeroNetwork(const std::string& path, const OptionsDict& /*options*/)
      : weights_(hero::LoadHeroWeights(path)) {
    capabilities_.input_format =
        pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2;
    capabilities_.output_format = pblczero::NetworkFormat::OUTPUT_WDL;
    capabilities_.moves_left = pblczero::NetworkFormat::MOVES_LEFT_NONE;
    CERR << "Hero net loaded: d=" << weights_.d << " layers=" << weights_.layers
         << " heads=" << weights_.heads << " dff=" << weights_.dff
         << " classes=" << weights_.classes;
  }
  const NetworkCapabilities& GetCapabilities() const override { return capabilities_; }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<HeroNetworkComputation>(this);
  }
  int GetMiniBatchSize() const override { return 256; }

 private:
  hero::HeroWeights weights_;
  NetworkCapabilities capabilities_;
};

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
