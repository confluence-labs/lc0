// Standalone loader check (no lc0 build needed):
//   g++ -std=c++17 -I src src/neural/hero/hero_weights{,_test}.cc -o /tmp/htwtest
//   /tmp/htwtest hero.htw
// Loads a .htw, asserts the HERO_1024/13 shape contract, prints a checksum to
// cross-check against the exporter.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "neural/hero/hero_weights.h"

using lczero::hero::HeroWeights;
using lczero::hero::LoadHeroWeights;

static int fails = 0;
static void check(bool ok, const char* what, long got = -1, long want = -1) {
  if (!ok) {
    std::printf("  FAIL %s (got %ld want %ld)\n", what, got, want);
    ++fails;
  }
}
static double sum(const std::vector<float>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: htwtest <file.htw>\n"); return 2; }
  HeroWeights w = LoadHeroWeights(argv[1]);

  std::printf("config: d=%d layers=%d heads=%d hd=%d dff=%d embed_dff=%d pol_d=%d classes=%d\n",
              w.d, w.layers, w.heads, w.hd, w.dff, w.embed_dff, w.pol_d, w.classes);
  check(w.d == 1024, "d", w.d, 1024);
  check(w.layers == 15, "layers", w.layers, 15);
  check(w.heads == 32, "heads", w.heads, 32);
  check(w.dff == 1280, "dff", w.dff, 1280);
  check(w.classes == 13, "classes", w.classes, 13);

  const int d = w.d, dff = w.dff, E = w.classes, pd = w.pol_d;
  check((long)w.embed_w.size() == (long)d * 240, "embed_w", w.embed_w.size(), (long)d * 240);
  check((long)w.preproc0_w.size() == 128L * 768, "preproc0", w.preproc0_w.size(), 128L * 768);
  check((int)w.layer.size() == w.layers, "n_layers", w.layer.size(), w.layers);

  long expert_params = 0;
  bool per_layer_ok = true;
  for (const auto& L : w.layer) {
    if ((long)L.q_w.size() != (long)d * d) per_layer_ok = false;
    if ((long)L.alpha.size() != (long)w.heads * 18) per_layer_ok = false;
    if ((long)L.free.size() != (long)w.heads * 64 * 64) per_layer_ok = false;
    if ((long)L.ffn_up.size() != (long)E * dff * d) per_layer_ok = false;
    if ((long)L.ffn_down.size() != (long)E * d * dff) per_layer_ok = false;
    if ((long)L.ln1_g.size() != d) per_layer_ok = false;
    expert_params += (long)L.ffn_up.size() + (long)L.ffn_down.size();
  }
  check(per_layer_ok, "per-layer shapes");
  check((long)w.pol_ppo_w.size() == 4L * pd, "pol_ppo", w.pol_ppo_w.size(), 4L * pd);
  check((long)w.val_v_w.size() == 3L * pd, "val_v", w.val_v_w.size(), 3L * pd);

  std::printf("expert-bank params: %ld (%.1f%% of a 579.3M net)\n",
              expert_params, 100.0 * expert_params / 579.3e6);
  std::printf("checksums (sum of fp32): embed_w=%.6f  L0.ffn_up=%.6f  L0.free=%.6f  val_v=%.6f\n",
              sum(w.embed_w), sum(w.layer[0].ffn_up), sum(w.layer[0].free), sum(w.val_v_w));

  std::printf(fails ? "\nLOADER TEST: %d FAIL\n" : "\nLOADER TEST: OK\n", fails);
  return fails ? 1 : 0;
}
