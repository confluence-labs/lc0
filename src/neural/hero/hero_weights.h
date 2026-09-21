// Hero net weights, parsed from a .htw side-file (see HERO_BACKEND.md).
// Mirrors network_legacy.h's MultiHeadWeights, but each encoder layer carries an
// E-way expert bank + a static attention bias instead of a single dense FFN.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lczero {
namespace hero {

using Vec = std::vector<float>;  // host weights are fp32 (cast to fp16/bf16 at GPU upload, as lc0 does)

struct HeroWeights {
  int d = 0, layers = 0, heads = 0, hd = 0, dff = 0, embed_dff = 0, pol_d = 0, classes = 0;

  // stem / embedding
  Vec preproc0_w;       // [128, 768]  factorised PE_DENSE
  Vec preproc1_w;       // [8192, 128]
  Vec embed_w;          // [d, 240]
  Vec embed_ln_g;       // [d]         LN gamma (beta == 0)
  Vec embed_up_w;       // [embed_dff, d]
  Vec embed_down_w;     // [d, embed_dff]
  Vec embed_ffn_ln_g;   // [d]

  struct Layer {
    Vec q_w, k_w, v_w, out_w;  // each [d, d], no bias
    Vec alpha;                 // [heads, 18]     static-bias mixing over geo masks
    Vec free;                  // [heads, 64, 64] free per-head bias table
    Vec ln1_g, ln2_g;          // [d]             LN gamma (beta == 0)
    Vec ffn_up;                // [classes, dff, d]
    Vec ffn_down;              // [classes, d, dff]
  };
  std::vector<Layer> layer;

  // heads (policy has biases; value is bias-free)
  Vec pol_embed_w, pol_embed_b;   // [pol_d, d], [pol_d]
  Vec pol_q_w, pol_q_b;           // [pol_d, pol_d], [pol_d]
  Vec pol_k_w, pol_k_b;           // [pol_d, pol_d], [pol_d]
  Vec pol_ppo_w;                  // [4, pol_d]
  Vec val_embed_w;                // [pol_d, d]
  Vec val_q_w, val_k_w;           // [pol_d, pol_d]
  // lc0bench: optional value-head biases (the played head has none; the value_q
  // aux head does). Empty => the forward passes nullptr, exactly as before.
  Vec val_embed_b, val_q_b, val_k_b;
  Vec val_v_w;                    // [3, pol_d]
};

// Parse a .htw file into HeroWeights (fp16 blobs -> fp32 vectors). Throws
// std::runtime_error on a bad magic / missing tensor / short read.
HeroWeights LoadHeroWeights(const std::string& path);

// True if the file begins with the HERW magic (for the loader's format sniff).
bool IsHeroWeightsFile(const std::string& path);

}  // namespace hero
}  // namespace lczero
