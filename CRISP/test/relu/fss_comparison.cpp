#include "fss_comparison.h"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <algorithm>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#else
#define OPENSSL_SUPPRESS_DEPRECATED
#include <openssl/aes.h>
#endif
namespace {
// ---------------------------------------------------------------------------
// AES-128 PRG: G(seed) = AES_seed(C_0) || AES_seed(C_1) || AES_seed(C_2) ||
// AES_seed(C_3), where the C_i are fixed, domain-separated 16-byte blocks.
// Only 50 of the 64 output bytes are used (two child seeds, two 64-bit value
// corrections, two control bits); the old code encrypted 80 bytes and threw
// away 30 of them.
//
// On AArch64 the round function uses the constant-time AES instructions. A
// fresh key schedule is needed for every expansion because the DCF seed *is*
// the AES key. Independent expansions are interleaved, which hides the serial
// dependency chain of the key schedule. The previous EVP implementation paid
// a provider lookup and context reset on every 80-byte call (about 370 ns), and
// OpenSSL 3's EVP initialisation is serialised across threads, which is why
// the four-thread benchmark showed no speedup at all.
// ---------------------------------------------------------------------------
alignas(16) const uint8_t BLK_D0[4][16]={{0,0},{0,1},{0,2},{0,3}};
alignas(16) const uint8_t BLK_D1[1][16]={{1,0}};
constexpr uint32_t RCON[11]={0,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};
uint32_t le32(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}

#if defined(__aarch64__)
struct AesKey{uint8x16_t rk[11];};
inline uint8x16_t prg_sub_rot(uint8x16_t k){
  // Broadcast the last word, apply SubBytes+ShiftRows (AESE with a zero key);
  // identical columns make ShiftRows a no-op, so every column is SubWord(w3).
  // RotWord is a constant table permutation of the four bytes in each word.
  uint32_t w3=vgetq_lane_u32(vreinterpretq_u32_u8(k),3);
  uint8x16_t b=vreinterpretq_u8_u32(vdupq_n_u32(w3));
  uint8x16_t s=vaeseq_u8(b,vdupq_n_u8(0));
  static const uint8_t idx[16]={1,2,3,0,5,6,7,4,9,10,11,8,13,14,15,12};
  return vqtbl1q_u8(s,vld1q_u8(idx));
}
inline uint8x16_t prg_next_rk(uint8x16_t k,uint32_t rc){
  uint8x16_t u=veorq_u8(prg_sub_rot(k),vreinterpretq_u8_u32(vdupq_n_u32(rc)));
  uint8x16_t p=k;
  p=veorq_u8(p,vextq_u8(vdupq_n_u8(0),p,12));
  p=veorq_u8(p,vextq_u8(vdupq_n_u8(0),p,8));
  return veorq_u8(p,u);
}
inline void prg_schedule(const Seed&s,AesKey&ks){
  uint8x16_t k=vld1q_u8(s.bytes.data());ks.rk[0]=k;
  for(int r=1;r<=10;r++){k=prg_next_rk(k,RCON[r]);ks.rk[r]=k;}
}
inline uint8x16_t prg_encrypt(uint8x16_t c,const AesKey&ks){
  const uint8x16_t z=vdupq_n_u8(0);
  uint8x16_t s=veorq_u8(c,ks.rk[0]);
  for(int r=1;r<10;r++)s=veorq_u8(vaesmcq_u8(vaeseq_u8(s,z)),ks.rk[r]);
  return veorq_u8(vaeseq_u8(s,z),ks.rk[10]);
}
template<int W>
inline void expand_group(const Seed* in,Expansion* out){
  AesKey ks[W];
  for(int j=0;j<W;j++)prg_schedule(in[j],ks[j]);
  uint8x16_t o[W][4];
  for(int b=0;b<4;b++){uint8x16_t c=vld1q_u8(BLK_D0[b]);
    for(int j=0;j<W;j++)o[j][b]=prg_encrypt(c,ks[j]);}
  for(int j=0;j<W;j++){
    alignas(16) uint8_t p[64];
    for(int b=0;b<4;b++)vst1q_u8(p+16*b,o[j][b]);
    std::memcpy(out[j].left.bytes.data(),p,16);
    std::memcpy(out[j].right.bytes.data(),p+16,16);
    out[j].vl={le32(p+32),le32(p+36)};
    out[j].vr={le32(p+48),le32(p+52)};
    out[j].tl=p[56]&1;out[j].tr=p[57]&1;
  }
}
template<int W>
inline void convert_group(const Seed* in,Z2NVec* out){
  AesKey ks[W];
  for(int j=0;j<W;j++)prg_schedule(in[j],ks[j]);
  uint8x16_t c=vld1q_u8(BLK_D1[0]);uint8x16_t o[W];
  for(int j=0;j<W;j++)o[j]=prg_encrypt(c,ks[j]);
  for(int j=0;j<W;j++){
    alignas(16) uint8_t p[16];vst1q_u8(p,o[j]);
    out[j]={le32(p),le32(p+4)};
  }
}
#else
// Portable fallback: the low-level OpenSSL entry points avoid EVP's per-call
// provider lookup (and its cross-thread serialisation) on non-AArch64 hosts.
struct AesKey{AES_KEY k;};
inline void prg_schedule(const Seed&s,AesKey&ks){AES_set_encrypt_key(s.bytes.data(),128,&ks.k);}
inline void prg_encrypt(const uint8_t in[16],uint8_t out[16],const AesKey&ks){AES_encrypt(in,out,&ks.k);}
template<int W>
inline void expand_group(const Seed* in,Expansion* out){
  AesKey ks[W];
  for(int j=0;j<W;j++)prg_schedule(in[j],ks[j]);
  for(int j=0;j<W;j++){
    uint8_t p[64];
    for(int b=0;b<4;b++)prg_encrypt(BLK_D0[b],p+16*b,ks[j]);
    std::memcpy(out[j].left.bytes.data(),p,16);
    std::memcpy(out[j].right.bytes.data(),p+16,16);
    out[j].vl={le32(p+32),le32(p+36)};
    out[j].vr={le32(p+48),le32(p+52)};
    out[j].tl=p[56]&1;out[j].tr=p[57]&1;
  }
}
template<int W>
inline void convert_group(const Seed* in,Z2NVec* out){
  AesKey ks[W];
  for(int j=0;j<W;j++)prg_schedule(in[j],ks[j]);
  for(int j=0;j<W;j++){
    uint8_t p[16];prg_encrypt(BLK_D1[0],p,ks[j]);
    out[j]={le32(p),le32(p+4)};
  }
}
#endif

// Reference implementation used only by prg_self_test(): the original
// user-keyed AES-128 ECB construction through EVP.
std::array<uint8_t,80> prg_reference(const Seed&s,uint32_t domain){
  struct Context{EVP_CIPHER_CTX*p=EVP_CIPHER_CTX_new();~Context(){EVP_CIPHER_CTX_free(p);}};
  thread_local Context holder;auto ctx=holder.p;
  if(!ctx)throw std::runtime_error("EVP allocation");
  std::array<uint8_t,80> in{},out{};
  for(unsigned i=0;i<5;++i){in[i*16]=uint8_t(domain);in[i*16+1]=uint8_t(i);}
  int len=0;
  if(EVP_EncryptInit_ex(ctx,EVP_aes_128_ecb(),nullptr,s.bytes.data(),nullptr)!=1 ||
     EVP_CIPHER_CTX_set_padding(ctx,0)!=1 ||
     EVP_EncryptUpdate(ctx,out.data(),&len,in.data(),in.size())!=1 ||len!=80)
    throw std::runtime_error("AES PRG failure");
  return out;
}
} // namespace

Seed operator^(const Seed&a,const Seed&b){Seed c;for(size_t i=0;i<16;++i)c.bytes[i]=a.bytes[i]^b.bytes[i];return c;}
Seed random_seed(){Seed s;if(RAND_bytes(s.bytes.data(),16)!=1)throw std::runtime_error("RAND_bytes");return s;}
uint32_t random_ring(){uint32_t x;if(RAND_bytes(reinterpret_cast<uint8_t*>(&x),4)!=1)throw std::runtime_error("RAND_bytes");return x;}
void random_seeds(Seed*out,size_t n){if(n&&RAND_bytes(reinterpret_cast<uint8_t*>(out),int(n*sizeof(Seed)))!=1)throw std::runtime_error("RAND_bytes");}
void random_rings(uint32_t*out,size_t n){if(n&&RAND_bytes(reinterpret_cast<uint8_t*>(out),int(n*sizeof(uint32_t)))!=1)throw std::runtime_error("RAND_bytes");}

Expansion expand_seed(const Seed&s){Expansion e;expand_group<1>(&s,&e);return e;}
Z2NVec convert_to_z2n_vec(const Seed&s){Z2NVec v;convert_group<1>(&s,&v);return v;}

void expand_seed_batch(const Seed* in,Expansion* out,size_t n){
  size_t i=0;
  for(;i+4<=n;i+=4)expand_group<4>(in+i,out+i);
  size_t r=n-i;
  if(r>=2){expand_group<2>(in+i,out+i);i+=2;r-=2;}
  if(r)expand_group<1>(in+i,out+i);
}
void convert_to_z2n_vec_batch(const Seed* in,Z2NVec* out,size_t n){
  size_t i=0;
  for(;i+4<=n;i+=4)convert_group<4>(in+i,out+i);
  size_t r=n-i;
  if(r>=2){convert_group<2>(in+i,out+i);i+=2;r-=2;}
  if(r)convert_group<1>(in+i,out+i);
}

namespace {
constexpr size_t MAX_BATCH=64;
// One DCF generation step shared by all batched keys. Keeping the whole batch
// in one loop lets expand_seed_batch interleave 2*n independent AES chains.
void gen_less_than_impl(const uint64_t* a,const Z2NVec* b,size_t n,
                        std::pair<FSS_Key,FSS_Key>* out){
  Seed s0[MAX_BATCH],s1[MAX_BATCH];
  uint8_t t0[MAX_BATCH],t1[MAX_BATCH];
  Z2NVec va[MAX_BATCH];
  Seed all[2*MAX_BATCH];
  Expansion ex[2*MAX_BATCH];
  for(size_t i=0;i<n;++i){
    s0[i]=random_seed();s1[i]=random_seed();
    out[i].first.seed=s0[i];out[i].second.seed=s1[i];
    t0[i]=0;t1[i]=1;va[i]={};
  }
  for(int i=0;i<32;++i){
    for(size_t j=0;j<n;++j){all[2*j]=s0[j];all[2*j+1]=s1[j];}
    expand_seed_batch(all,ex,2*n);
    for(size_t j=0;j<n;++j){
      const Expansion&o0=ex[2*j];const Expansion&o1=ex[2*j+1];
      uint8_t bit=uint8_t((a[j]>>(31-i))&1);CWElement cw;
      cw.t_cw_L=o0.tl^o1.tl^bit^1;cw.t_cw_R=o0.tr^o1.tr^bit;
      uint64_t sguid=t1[j]?uint64_t(-1):1;
      if(!bit){
        cw.s_cw=o0.right^o1.right;
        cw.v_cw=dot_product_z2n_vec(sub_z2n_vec(sub_z2n_vec(o1.vr,o0.vr),va[j]),sguid);
        va[j]=add_z2n_vec(va[j],sub_z2n_vec(o0.vl,o1.vl));
      }else{
        cw.s_cw=o0.left^o1.left;
        cw.v_cw=dot_product_z2n_vec(add_z2n_vec(sub_z2n_vec(sub_z2n_vec(o1.vl,o0.vl),va[j]),b[j]),sguid);
        va[j]=add_z2n_vec(va[j],sub_z2n_vec(o0.vr,o1.vr));
      }
      va[j]=add_z2n_vec(va[j],dot_product_z2n_vec(cw.v_cw,sguid));
      Seed a0=bit?o0.right:o0.left,a1=bit?o1.right:o1.left;
      s0[j]=t0[j]?a0^cw.s_cw:a0;s1[j]=t1[j]?a1^cw.s_cw:a1;
      t0[j]=uint8_t((bit?o0.tr:o0.tl)^(t0[j]*(bit?cw.t_cw_R:cw.t_cw_L)));
      t1[j]=uint8_t((bit?o1.tr:o1.tl)^(t1[j]*(bit?cw.t_cw_R:cw.t_cw_L)));
      out[j].first.CW[i]=cw;out[j].second.CW[i]=cw;
    }
  }
  for(size_t j=0;j<n;++j){all[2*j]=s0[j];all[2*j+1]=s1[j];}
  Z2NVec fin[2*MAX_BATCH];
  convert_to_z2n_vec_batch(all,fin,2*n);
  for(size_t j=0;j<n;++j){
    Z2NVec fc=dot_product_z2n_vec(sub_z2n_vec(sub_z2n_vec(fin[2*j+1],fin[2*j]),va[j]),t1[j]?uint64_t(-1):1);
    out[j].first.final_cw=fc;out[j].second.final_cw=fc;
  }
}
void eval_less_than_impl(uint8_t party,const FSS_Key* const* keys,const uint64_t* x,size_t n,Z2NVec* out){
  Seed s[MAX_BATCH];uint8_t t[MAX_BATCH];Z2NVec v[MAX_BATCH];
  Expansion ex[MAX_BATCH];Z2NVec fin[MAX_BATCH];
  const uint64_t sign=party?uint64_t(-1):1;
  for(size_t j=0;j<n;++j){s[j]=keys[j]->seed;t[j]=party;v[j]={};}
  for(int i=0;i<32;++i){
    expand_seed_batch(s,ex,n);
    for(size_t j=0;j<n;++j){
      bool bit=(x[j]>>(31-i))&1;const CWElement&cw=keys[j]->CW[i];
      v[j]=add_z2n_vec(v[j],dot_product_z2n_vec(
            add_z2n_vec(bit?ex[j].vr:ex[j].vl,dot_product_z2n_vec(cw.v_cw,t[j])),sign));
      Seed next=bit?ex[j].right:ex[j].left;
      s[j]=t[j]?next^cw.s_cw:next;
      t[j]=uint8_t((bit?ex[j].tr:ex[j].tl)^(t[j]*(bit?cw.t_cw_R:cw.t_cw_L)));
    }
  }
  convert_to_z2n_vec_batch(s,fin,n);
  for(size_t j=0;j<n;++j)
    out[j]=add_z2n_vec(v[j],dot_product_z2n_vec(
            add_z2n_vec(fin[j],dot_product_z2n_vec(keys[j]->final_cw,t[j])),sign));
}
} // namespace

std::pair<FSS_Key,FSS_Key> FssComparison::gen_less_than(uint64_t a,Z2NVec b){
  std::pair<FSS_Key,FSS_Key> out;
  gen_less_than_impl(&a,&b,1,&out);
  return out;
}
Z2NVec FssComparison::eval_less_than(uint8_t party,const FSS_Key&k,uint64_t x)const{
  if(party>1)throw std::invalid_argument("party must be 0 or 1");
  const FSS_Key*p=&k;Z2NVec out;
  eval_less_than_impl(party,&p,&x,1,&out);
  return out;
}
void gen_less_than_batch(const uint64_t* a,const Z2NVec* b,size_t n,std::pair<FSS_Key,FSS_Key>* out){
  while(n){size_t m=n<MAX_BATCH?n:MAX_BATCH;gen_less_than_impl(a,b,m,out);a+=m;b+=m;out+=m;n-=m;}
}
void eval_less_than_batch(uint8_t party,const FSS_Key* const* keys,const uint64_t* x,size_t n,Z2NVec* out){
  if(party>1)throw std::invalid_argument("party must be 0 or 1");
  while(n){size_t m=n<MAX_BATCH?n:MAX_BATCH;eval_less_than_impl(party,keys,x,m,out);keys+=m;x+=m;out+=m;n-=m;}
}

bool prg_self_test(){
  Seed seeds[8];random_seeds(seeds,8);
  for(int i=0;i<8;++i){
    auto ref=prg_reference(seeds[i],0);
    Expansion fast=expand_seed(seeds[i]);
    if(std::memcmp(fast.left.bytes.data(),ref.data(),16)!=0)return false;
    if(std::memcmp(fast.right.bytes.data(),ref.data()+16,16)!=0)return false;
    if(fast.vl[0]!=le32(ref.data()+32)||fast.vl[1]!=le32(ref.data()+36))return false;
    if(fast.vr[0]!=le32(ref.data()+48)||fast.vr[1]!=le32(ref.data()+52))return false;
    if(fast.tl!=(ref[56]&1)||fast.tr!=(ref[57]&1))return false;
    auto cref=prg_reference(seeds[i],1);
    Z2NVec conv=convert_to_z2n_vec(seeds[i]);
    if(conv[0]!=le32(cref.data())||conv[1]!=le32(cref.data()+4))return false;
  }
  return true;
}

void put32(std::vector<uint8_t>&o,uint32_t x){for(int j=0;j<4;++j)o.push_back(uint8_t(x>>(8*j)));}
uint32_t get32(const uint8_t*&p,const uint8_t*e){if(e-p<4)throw std::runtime_error("truncated key");auto v=le32(p);p+=4;return v;}
void write_key(std::vector<uint8_t>&o,const FSS_Key&k){o.insert(o.end(),k.seed.bytes.begin(),k.seed.bytes.end());for(auto&c:k.CW){o.insert(o.end(),c.s_cw.bytes.begin(),c.s_cw.bytes.end());for(auto v:c.v_cw)put32(o,v);o.push_back(c.t_cw_L);o.push_back(c.t_cw_R);}for(auto v:k.final_cw)put32(o,v);}
FSS_Key read_key(const uint8_t*&p,const uint8_t*e){if(size_t(e-p)<FSS_KEY_BYTES)throw std::runtime_error("truncated key");FSS_Key k;std::copy_n(p,16,k.seed.bytes.data());p+=16;for(auto&c:k.CW){std::copy_n(p,16,c.s_cw.bytes.data());p+=16;for(auto&v:c.v_cw)v=get32(p,e);c.t_cw_L=*p++;c.t_cw_R=*p++;if(c.t_cw_L>1||c.t_cw_R>1)throw std::runtime_error("bad control bit");}for(auto&v:k.final_cw)v=get32(p,e);return k;}
