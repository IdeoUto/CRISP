#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <utility>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
// The arithmetic ring and the PRG seed are deliberately separate types.
using Z2NVec = std::array<uint64_t,2>;
constexpr uint64_t N=uint64_t(1)<<32, mask=N-1;
struct Seed { std::array<uint8_t,16> bytes{}; };
Seed operator^(const Seed&,const Seed&);
Seed random_seed();
uint32_t random_ring();
// Bulk CSPRNG helpers: one RAND_bytes call for a whole batch instead of one
// call per seed/mask. They fill exactly n entries.
void random_seeds(Seed* out,size_t n);
void random_rings(uint32_t* out,size_t n);
Z2NVec convert_to_z2n_vec(const Seed&);
inline Z2NVec add_z2n_vec(const Z2NVec&a,const Z2NVec&b){return {(a[0]+b[0])&mask,(a[1]+b[1])&mask};}
inline Z2NVec sub_z2n_vec(const Z2NVec&a,const Z2NVec&b){return {(a[0]-b[0])&mask,(a[1]-b[1])&mask};}
inline Z2NVec dot_product_z2n_vec(const Z2NVec&b,uint64_t a){
#if defined(__aarch64__)
 // Narrow BOTH uint64 lanes, rather than selecting the low half of lane zero.
 uint32x2_t x=vmovn_u64(vld1q_u64(b.data()));
 uint64x2_t y=vmull_u32(x,vdup_n_u32(uint32_t(a)));
 Z2NVec out;vst1q_u64(out.data(),y);out[0]&=mask;out[1]&=mask;return out;
#else
 return {(b[0]*a)&mask,(b[1]*a)&mask};
#endif
}
struct Expansion{Seed left,right; Z2NVec vl,vr; uint8_t tl,tr;};
Expansion expand_seed(const Seed&);
// Vectorised PRG: expands n independent seeds. Independent chains are
// interleaved in groups of four, which hides the ten-round AES latency.
// n may be any size; the tail is handled with narrower groups.
void expand_seed_batch(const Seed* in,Expansion* out,size_t n);
void convert_to_z2n_vec_batch(const Seed* in,Z2NVec* out,size_t n);
struct CWElement{Seed s_cw;Z2NVec v_cw{};uint8_t t_cw_L=0,t_cw_R=0;};
struct FSS_Key{Seed seed;std::array<CWElement,32> CW;Z2NVec final_cw{};};
class FssComparison{
public:
 explicit FssComparison(uint8_t bits=32){if(bits!=32)throw std::invalid_argument("only Z_(2^32) supported");}
 std::pair<FSS_Key,FSS_Key> gen_less_than(uint64_t a,Z2NVec b);
 Z2NVec eval_less_than(uint8_t party,const FSS_Key&key,uint64_t x) const;
};
// Batched DCF generation/evaluation. These produce bit-for-bit the same
// interface as the scalar calls but share AES key schedules and interleave
// independent chains. Inputs are arrays of length n.
void gen_less_than_batch(const uint64_t* a,const Z2NVec* b,size_t n,
                         std::pair<FSS_Key,FSS_Key>* out);
void eval_less_than_batch(uint8_t party,const FSS_Key* const* keys,
                          const uint64_t* x,size_t n,Z2NVec* out);
// Checks the fast AES-128 PRG against an independent OpenSSL EVP computation.
bool prg_self_test();
// Canonical little-endian wire format, no struct padding or pointers.
constexpr size_t FSS_KEY_BYTES=16+32*(16+8+2)+8;
void write_key(std::vector<uint8_t>&out,const FSS_Key&key);
FSS_Key read_key(const uint8_t*&p,const uint8_t*end);
void put32(std::vector<uint8_t>&out,uint32_t v);
uint32_t get32(const uint8_t*&p,const uint8_t*end);
