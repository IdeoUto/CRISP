#include "relu.h"
namespace {
Z2NVec random_vec(){return {random_ring(),random_ring()};}
void write_relu(std::vector<uint8_t>&o,const Key&k){for(int j=0;j<int(k.mode);++j)write_key(o,k.dcf[j]);put32(o,k.r);for(auto v:k.b)put32(o,v);}
Key read_relu(const uint8_t*&p,const uint8_t*e,ReluMode m){Key k;k.mode=m;for(int j=0;j<int(m);++j)k.dcf[j]=read_key(p,e);k.r=get32(p,e);for(auto&v:k.b)v=get32(p,e);return k;}
}
std::pair<Key,Key> ReLUFunction::genReLU(uint32_t multiplier)const{
 Z2NVec beta={multiplier,uint32_t(0-uint64_t(multiplier)*r_)};FssComparison f;
 Key a,b;a.mode=b.mode=mode_;auto low=f.gen_less_than(r_,dot_product_z2n_vec(beta,uint64_t(-1)));a.dcf[0]=low.first;b.dcf[0]=low.second;
 Z2NVec constant=beta;
 if(mode_==ReluMode::exact){uint32_t end=r_+uint32_t(1u<<31);auto high=f.gen_less_than(end,beta);a.dcf[1]=high.first;b.dcf[1]=high.second;constant=end<r_?beta:Z2NVec{};}
 a.r=random_ring();b.r=r_-a.r;a.b=random_vec();b.b=sub_z2n_vec(constant,a.b);return {a,b};
}
uint32_t ReLUFunction::evalReLU(uint8_t party,const Key&k,uint32_t z)const{FssComparison f;auto b=k.b;for(int j=0;j<int(k.mode);++j)b=add_z2n_vec(b,f.eval_less_than(party,k.dcf[j],z));return uint32_t(b[0]*z+b[1]);}

// ---------------------------------------------------------------------------
// Batched authenticated ReLU.
//
// A single ReLU needs D DCFs for the value and D for the MAC (D=1 in the
// paper mode, D=2 exact). Evaluating those 2D trees one after another leaves
// the AES pipeline idle for most of each round. Here all 2D trees of a batch
// are flattened into one eval_less_than_batch call, so every AES key schedule
// is overlapped with three others.
// ---------------------------------------------------------------------------
void generate_authenticated_relu_batch(uint32_t mu,ReluMode mode,size_t n,
                                       std::vector<AuthKey>& party0,
                                       std::vector<AuthKey>& party1){
 const bool exact=(mode==ReluMode::exact);
 party0.resize(n);party1.resize(n);
 if(!n)return;
 std::vector<uint32_t> r(n);random_rings(r.data(),n);
 std::vector<uint64_t> alpha(n),endv(n);
 std::vector<Z2NVec> lowb_v(n),lowb_m(n),highb_v(n),highb_m(n),const_v(n),const_m(n);
 for(size_t i=0;i<n;++i){
  uint32_t ri=r[i];
  Z2NVec betav={1,uint32_t(0-uint64_t(ri))};
  Z2NVec betam={mu,uint32_t(0-uint64_t(mu)*ri)};
  alpha[i]=ri;
  // b for f_<_{alpha,beta} is -beta evaluated at the lower interval end.
  lowb_v[i]={uint32_t(0-1u),ri};
  lowb_m[i]={uint32_t(0-mu),uint32_t(uint64_t(mu)*ri)};
  const_v[i]=betav;const_m[i]=betam;
  if(exact){
   uint32_t end=ri+uint32_t(1u<<31);endv[i]=end;
   highb_v[i]=betav;highb_m[i]=betam;
   // end<r means the interval wrapped past 2^32; the constant remainder is
   // then beta, otherwise zero.
   if(!(end<ri)){const_v[i]={};const_m[i]={};}
  }
 }
 std::vector<std::pair<FSS_Key,FSS_Key>> low_v(n),low_m(n),high_v,high_m;
 gen_less_than_batch(alpha.data(),lowb_v.data(),n,low_v.data());
 gen_less_than_batch(alpha.data(),lowb_m.data(),n,low_m.data());
 if(exact){
  high_v.resize(n);high_m.resize(n);
  gen_less_than_batch(endv.data(),highb_v.data(),n,high_v.data());
  gen_less_than_batch(endv.data(),highb_m.data(),n,high_m.data());
 }
 // Correlated randomness: r = ra+rb, mu = mua+mub, mu*r = mura+murb, and the
 // value/MAC masks share the same r per party (checked by read_auth_key).
 std::vector<uint32_t> ra(n),mua(n),mura(n),rnd(4*n);
 random_rings(ra.data(),n);random_rings(mua.data(),n);random_rings(mura.data(),n);
 random_rings(rnd.data(),4*n);
 for(size_t i=0;i<n;++i){
  AuthKey&A=party0[i],&B=party1[i];
  A.value.mode=B.value.mode=A.mac.mode=B.mac.mode=mode;
  A.value.dcf[0]=low_v[i].first;B.value.dcf[0]=low_v[i].second;
  A.mac.dcf[0]=low_m[i].first;B.mac.dcf[0]=low_m[i].second;
  if(exact){
   A.value.dcf[1]=high_v[i].first;B.value.dcf[1]=high_v[i].second;
   A.mac.dcf[1]=high_m[i].first;B.mac.dcf[1]=high_m[i].second;
  }
  A.value.r=ra[i];B.value.r=r[i]-ra[i];
  A.mac.r=A.value.r;B.mac.r=B.value.r;
  Z2NVec ba_v={rnd[4*i],rnd[4*i+1]},ba_m={rnd[4*i+2],rnd[4*i+3]};
  A.value.b=ba_v;B.value.b=sub_z2n_vec(const_v[i],ba_v);
  A.mac.b=ba_m;B.mac.b=sub_z2n_vec(const_m[i],ba_m);
  A.mu_share=mua[i];B.mu_share=mu-mua[i];
  A.mur_share=mura[i];B.mur_share=uint32_t(uint64_t(mu)*r[i])-mura[i];
 }
}
void evaluate_authenticated_relu_batch(uint8_t party,const AuthKey* keys,
                                       const uint32_t* opened,size_t n,
                                       AuthOutput* out){
 constexpr size_t CHUNK=64;
 for(size_t base=0;base<n;base+=CHUNK){
  size_t m=n-base<CHUNK?n-base:CHUNK;
  const int d=int(keys[base].value.mode);
  std::array<const FSS_Key*,2*CHUNK*2> ptrs{};
  std::array<uint64_t,2*CHUNK*2> xs{};
  std::array<Z2NVec,2*CHUNK*2> res{};
  size_t t=0;
  for(size_t i=0;i<m;++i){
   for(int j=0;j<d;++j){ptrs[t]=&keys[base+i].value.dcf[j];xs[t]=opened[base+i];++t;}
   for(int j=0;j<d;++j){ptrs[t]=&keys[base+i].mac.dcf[j];xs[t]=opened[base+i];++t;}
  }
  eval_less_than_batch(party,ptrs.data(),xs.data(),t,res.data());
  t=0;
  for(size_t i=0;i<m;++i){
   Z2NVec bv=keys[base+i].value.b,bm=keys[base+i].mac.b;
   for(int j=0;j<d;++j)bv=add_z2n_vec(bv,res[t++]);
   for(int j=0;j<d;++j)bm=add_z2n_vec(bm,res[t++]);
   uint64_t z=opened[base+i];
   out[base+i].value=uint32_t(bv[0]*z+bv[1]);
   out[base+i].mac=uint32_t(bm[0]*z+bm[1]);
   out[base+i].input_mac=uint32_t(uint64_t(keys[base+i].mu_share)*z-keys[base+i].mur_share);
  }
 }
}
std::pair<AuthKey,AuthKey> generate_authenticated_relu(uint32_t mu,ReluMode mode){
 std::vector<AuthKey> a,b;generate_authenticated_relu_batch(mu,mode,1,a,b);
 return {std::move(a[0]),std::move(b[0])};
}
AuthOutput evaluate_authenticated_relu(uint8_t party,const AuthKey&k,uint32_t z){
 AuthOutput o;evaluate_authenticated_relu_batch(party,&k,&z,1,&o);return o;
}
size_t relu_key_bytes(ReluMode m){return size_t(m)*FSS_KEY_BYTES+12;}
size_t auth_key_bytes(ReluMode m){return 2*relu_key_bytes(m)+8;}
void write_auth_key(std::vector<uint8_t>&o,const AuthKey&k){write_relu(o,k.value);write_relu(o,k.mac);put32(o,k.mu_share);put32(o,k.mur_share);}
AuthKey read_auth_key(const uint8_t*&p,const uint8_t*e,ReluMode m){AuthKey k;k.value=read_relu(p,e,m);k.mac=read_relu(p,e,m);k.mu_share=get32(p,e);k.mur_share=get32(p,e);if(k.value.r!=k.mac.r)throw std::runtime_error("inconsistent mask shares");return k;}
