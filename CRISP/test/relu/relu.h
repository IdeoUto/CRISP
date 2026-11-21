#pragma once
#include "fss_comparison.h"
// paper_approx implements Algorithms 1/2 verbatim; exact uses the circular
// interval [r,r+2^31) and two DCFs, fixing modular-wrap errors.
enum class ReluMode:uint8_t{paper_approx=1,exact=2};
struct Key{std::array<FSS_Key,2> dcf;uint32_t r=0;Z2NVec b{};ReluMode mode=ReluMode::exact;};
class ReLUFunction{
 uint32_t r_;ReluMode mode_;
public:
 ReLUFunction(uint64_t r,uint64_t modulus,ReluMode mode=ReluMode::exact):r_(r),mode_(mode){if(modulus!=N)throw std::invalid_argument("ring must be 2^32");}
 std::pair<Key,Key> genReLU(uint32_t multiplier=1)const;
 uint32_t evalReLU(uint8_t party,const Key&key,uint32_t opened)const;
};
struct AuthKey{Key value,mac;uint32_t mu_share=0,mur_share=0;};
struct AuthOutput{uint32_t value,mac,input_mac;};
std::pair<AuthKey,AuthKey> generate_authenticated_relu(uint32_t mu,ReluMode mode);
AuthOutput evaluate_authenticated_relu(uint8_t party,const AuthKey&k,uint32_t opened);
// Batched authenticated ReLU. The value and MAC DCFs of every element are
// evaluated in one interleaved PRG batch, which is what makes the AES latency
// hiding effective. Outputs are bit-for-bit identical to the scalar calls.
void generate_authenticated_relu_batch(uint32_t mu,ReluMode mode,size_t n,
                                       std::vector<AuthKey>& party0,
                                       std::vector<AuthKey>& party1);
void evaluate_authenticated_relu_batch(uint8_t party,const AuthKey* keys,
                                       const uint32_t* opened,size_t n,
                                       AuthOutput* out);
size_t relu_key_bytes(ReluMode mode);
size_t auth_key_bytes(ReluMode mode);
void write_auth_key(std::vector<uint8_t>&out,const AuthKey&key);
AuthKey read_auth_key(const uint8_t*&p,const uint8_t*end,ReluMode mode);
