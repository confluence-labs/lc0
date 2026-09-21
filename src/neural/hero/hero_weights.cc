#include "neural/hero/hero_weights.h"

#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace lczero {
namespace hero {

namespace {

constexpr uint32_t kMagic = 0x48455257;  // 'HERW'
constexpr uint64_t kPage = 4096;

float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;  // +/- zero
    } else {     // subnormal -> normalized float
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
      mant &= 0x3FF;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

template <typename T>
T Read(std::ifstream& f) {
  T v;
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!f) throw std::runtime_error("hero .htw: short read");
  return v;
}

struct Entry {
  std::string name;
  uint32_t ndim;
  uint32_t shape[6];
  uint64_t offset;
  uint64_t nbytes;
};

}  // namespace

bool IsHeroWeightsFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t magic = 0;
  f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  return f && magic == kMagic;
}

HeroWeights LoadHeroWeights(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("hero .htw: cannot open " + path);

  if (Read<uint32_t>(f) != kMagic) throw std::runtime_error("hero .htw: bad magic");
  const uint32_t version = Read<uint32_t>(f);
  const uint32_t dtype = Read<uint32_t>(f);
  if (version != 1 || dtype != 1) {
    throw std::runtime_error("hero .htw: unsupported version/dtype");
  }

  HeroWeights w;
  w.d = Read<uint32_t>(f);
  w.layers = Read<uint32_t>(f);
  w.heads = Read<uint32_t>(f);
  w.hd = Read<uint32_t>(f);
  w.dff = Read<uint32_t>(f);
  w.embed_dff = Read<uint32_t>(f);
  w.pol_d = Read<uint32_t>(f);
  w.classes = Read<uint32_t>(f);
  w.layer.resize(w.layers);

  const uint32_t n = Read<uint32_t>(f);
  std::vector<Entry> dir(n);
  for (auto& e : dir) {
    char name[48];
    f.read(name, sizeof(name));
    if (!f) throw std::runtime_error("hero .htw: short read (dir)");
    e.name.assign(name, strnlen(name, sizeof(name)));
    e.ndim = Read<uint32_t>(f);
    for (int i = 0; i < 6; ++i) e.shape[i] = Read<uint32_t>(f);
    e.offset = Read<uint64_t>(f);
    e.nbytes = Read<uint64_t>(f);
  }

  // blob section begins at the first page boundary after the directory
  const uint64_t hdr_end = f.tellg();
  const uint64_t blob_base = (hdr_end + kPage - 1) / kPage * kPage;

  // name -> destination vector. Per-layer entries handled separately below.
  std::map<std::string, Vec*> dst = {
      {"preproc.0.weight", &w.preproc0_w},   {"preproc.1.weight", &w.preproc1_w},
      {"embed.weight", &w.embed_w},          {"embed_ln.weight", &w.embed_ln_g},
      {"embed_up.weight", &w.embed_up_w},    {"embed_down.weight", &w.embed_down_w},
      {"embed_ffn_ln.weight", &w.embed_ffn_ln_g},
      {"policy.embed.weight", &w.pol_embed_w}, {"policy.embed.bias", &w.pol_embed_b},
      {"policy.q.weight", &w.pol_q_w},       {"policy.q.bias", &w.pol_q_b},
      {"policy.k.weight", &w.pol_k_w},       {"policy.k.bias", &w.pol_k_b},
      {"policy.ppo.weight", &w.pol_ppo_w},   {"value.embed.weight", &w.val_embed_w},
      {"value.q.weight", &w.val_q_w},        {"value.k.weight", &w.val_k_w},
      {"value.v.weight", &w.val_v_w},
      {"value.embed.bias", &w.val_embed_b},  // lc0bench: optional, see patch 0002
      {"value.q.bias", &w.val_q_b},          {"value.k.bias", &w.val_k_b},
  };

  auto layer_dst = [&](const std::string& name) -> Vec* {
    // "layers.{i}.{field}"
    if (name.rfind("layers.", 0) != 0) return nullptr;
    const size_t dot = name.find('.', 7);
    if (dot == std::string::npos) return nullptr;
    const int i = std::stoi(name.substr(7, dot - 7));
    if (i < 0 || i >= w.layers) return nullptr;
    const std::string field = name.substr(dot + 1);
    auto& L = w.layer[i];
    if (field == "attn.q.weight") return &L.q_w;
    if (field == "attn.k.weight") return &L.k_w;
    if (field == "attn.v.weight") return &L.v_w;
    if (field == "attn.out.weight") return &L.out_w;
    if (field == "attn.alpha") return &L.alpha;
    if (field == "attn.free") return &L.free;
    if (field == "ln1.weight") return &L.ln1_g;
    if (field == "ln2.weight") return &L.ln2_g;
    if (field == "ffn.up") return &L.ffn_up;
    if (field == "ffn.down") return &L.ffn_down;
    return nullptr;
  };

  std::vector<uint16_t> raw;
  for (const auto& e : dir) {
    Vec* v = dst.count(e.name) ? dst[e.name] : layer_dst(e.name);
    if (!v) throw std::runtime_error("hero .htw: unexpected tensor " + e.name);
    const uint64_t count = e.nbytes / 2;
    raw.resize(count);
    f.seekg(blob_base + e.offset);
    f.read(reinterpret_cast<char*>(raw.data()), e.nbytes);
    if (!f) throw std::runtime_error("hero .htw: short read (blob " + e.name + ")");
    v->resize(count);
    for (uint64_t i = 0; i < count; ++i) (*v)[i] = HalfToFloat(raw[i]);
  }
  return w;
}

}  // namespace hero
}  // namespace lczero
