#pragma once
#include <seal/seal.h>
#include <complex>
#include <vector>
#include <map>
#include <memory>
namespace crisp {
using C=std::complex<double>;
size_t pow2(size_t n);
struct Parameters{size_t degree=8192;double scale=1099511627776.0;std::vector<int>bits{60,40,40,60};seal::EncryptionParameters make()const;};
struct FCPlan{
 size_t rows,cols,slots,L,P,D;std::vector<seal::Plaintext>diagonals;std::vector<uint8_t>active;
 FCPlan(size_t rows,size_t cols,const std::vector<double>&weights,seal::CKKSEncoder&,double scale);
 std::vector<C> pack(const std::vector<C>&x)const;
 std::vector<C> unpack(const std::vector<C>&x)const;
 seal::Ciphertext eval(const seal::Ciphertext&,seal::Evaluator&,const seal::GaloisKeys&,seal::Encryptor&)const;
 std::vector<int> rotations()const;
};
struct ConvShape{size_t h,w,ci,co,k,pad,stride=1;size_t oh()const{return (h+2*pad-k)/stride+1;}size_t ow()const{return (w+2*pad-k)/stride+1;}};
struct ConvPlan{
 ConvShape shape;size_t slots,L,P,inputs,outputs;double scale;
 std::vector<double>weights;
 ConvPlan(ConvShape,const std::vector<double>&,seal::CKKSEncoder&,double scale);
 std::vector<std::vector<C>> pack(const std::vector<C>&x)const;
 std::vector<C> unpack(const std::vector<std::vector<C>>&x)const;
 std::vector<int> rotations()const;
 // Correct CKKS channel-diagonal output packing. Rotations of each input are
 // reused for every output-channel group. Only one rescale per output.
 std::vector<seal::Ciphertext> eval(const std::vector<seal::Ciphertext>&,seal::Evaluator&,seal::CKKSEncoder&,const seal::GaloisKeys&,int threads,seal::Encryptor&)const;
};
std::vector<C> plain_fc(size_t rows,size_t cols,const std::vector<double>&w,const std::vector<C>&x);
std::vector<C> plain_conv(ConvShape,const std::vector<double>&w,const std::vector<C>&x);
}
