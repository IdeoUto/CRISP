#include "relu.h"
#include <iostream>
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(){try{
 require(dot_product_z2n_vec({3,5},7)==Z2NVec{21,35},"NEON lanes");FssComparison f;
 for(uint32_t a:{0u,1u,100u,0x7fffffffu,0x80000000u,0xffffffffu}){
 auto beta=Z2NVec{random_ring(),random_ring()};auto keys=f.gen_less_than(a,beta);
 std::vector<uint8_t> wire;write_key(wire,keys.first);require(wire.size()==FSS_KEY_BYTES,"wire size");bool rejected=false;try{const uint8_t*bad=wire.data();read_key(bad,bad+wire.size()-1);}catch(const std::runtime_error&){rejected=true;}require(rejected,"truncation must reject");auto p=wire.data();const uint8_t*q=p;auto copy=read_key(q,p+wire.size());
 for(unsigned i=0;i<100;++i){uint32_t x=i<4?uint32_t(a+i-2):random_ring();auto y=add_z2n_vec(f.eval_less_than(0,copy,x),f.eval_less_than(1,keys.second,x));require(y==(x<a?beta:Z2NVec{}),"DCF correctness");}}
 for(auto mode:{ReluMode::paper_approx,ReluMode::exact})for(int iter=0;iter<40;++iter){uint32_t mu=random_ring();auto ks=generate_authenticated_relu(mu,mode);std::vector<uint8_t>w;write_auth_key(w,ks.second);require(w.size()==auth_key_bytes(mode),"auth size");const uint8_t*p=w.data();auto k1=read_auth_key(p,p+w.size(),mode);
 for(uint32_t x:{0u,1u,100u,0xffffffffu,0xffffff9bu,0x80000000u,0x7fffffffu,random_ring()}){uint32_t r=ks.first.value.r+k1.value.r,z=x+r;auto a=evaluate_authenticated_relu(0,ks.first,z),b=evaluate_authenticated_relu(1,k1,z);uint32_t expected=mode==ReluMode::exact?(x<0x80000000u?x:0):(z>=r?x:0);require(uint32_t(a.value+b.value)==expected,"ReLU");require(uint32_t(a.mac+b.mac)==uint32_t(uint64_t(mu)*expected),"output MAC");require(uint32_t(a.input_mac+b.input_mac)==uint32_t(uint64_t(mu)*x),"input MAC");}}
 // Public boundary demonstrates why the paper approximation is not exact.
 ReLUFunction g(100,N,ReluMode::paper_approx);auto k=g.genReLU();uint32_t x=uint32_t(-101),z=x+100;require(uint32_t(g.evalReLU(0,k.first,z)+g.evalReLU(1,k.second,z))==x,"documented wrap behavior");
 // Fast PRG must reproduce the independent EVP construction bit for bit.
 require(prg_self_test(),"PRG self-test");
 // Batched DCF evaluation must agree with the scalar evaluator.
 {FssComparison f;std::vector<FSS_Key> k0(9),k1(9);std::vector<uint64_t> xs(9);std::vector<Z2NVec> beta(9);
  for(int i=0;i<9;++i){uint64_t a=uint32_t(i*0x9e3779b1u);beta[i]={random_ring(),random_ring()};auto p=f.gen_less_than(a,beta[i]);k0[i]=p.first;k1[i]=p.second;xs[i]=i<3?a+i:random_ring();}
  std::vector<const FSS_Key*> p0(9),p1(9);for(int i=0;i<9;++i){p0[i]=&k0[i];p1[i]=&k1[i];}
  std::vector<Z2NVec> o0(9),o1(9);eval_less_than_batch(0,p0.data(),xs.data(),9,o0.data());eval_less_than_batch(1,p1.data(),xs.data(),9,o1.data());
  for(int i=0;i<9;++i){auto scalar=add_z2n_vec(f.eval_less_than(0,k0[i],xs[i]),f.eval_less_than(1,k1[i],xs[i]));require(add_z2n_vec(o0[i],o1[i])==scalar,"batched DCF");require(scalar==(xs[i]<uint32_t(i*0x9e3779b1u)?beta[i]:Z2NVec{}),"batched DCF semantics");}}
 // Batched authenticated ReLU: same interface, all shares must reconstruct.
 for(auto mode:{ReluMode::paper_approx,ReluMode::exact}){
  uint32_t mu=random_ring();size_t n=23;std::vector<AuthKey> a,b;
  generate_authenticated_relu_batch(mu,mode,n,a,b);require(a.size()==n&&b.size()==n,"batch size");
  std::vector<uint32_t> xs(n),zs(n),expected(n);
  for(size_t i=0;i<n;++i){xs[i]=i%4==0?random_ring():uint32_t(i*7919);zs[i]=xs[i]+a[i].value.r+b[i].value.r;
   expected[i]=mode==ReluMode::exact?(xs[i]<0x80000000u?xs[i]:0):(zs[i]>=a[i].value.r+b[i].value.r?xs[i]:0);
   AuthOutput oa=evaluate_authenticated_relu(0,a[i],zs[i]),ob=evaluate_authenticated_relu(1,b[i],zs[i]);
   require(uint32_t(oa.value+ob.value)==expected[i],"batch ReLU");
   require(uint32_t(oa.mac+ob.mac)==uint32_t(uint64_t(mu)*expected[i]),"batch output MAC");
   require(uint32_t(oa.input_mac+ob.input_mac)==uint32_t(uint64_t(mu)*xs[i]),"batch input MAC");}
  std::vector<AuthOutput> outs(n);evaluate_authenticated_relu_batch(0,a.data(),zs.data(),n,outs.data());
  for(size_t i=0;i<n;++i){auto ob=evaluate_authenticated_relu(1,b[i],zs[i]);
   require(uint32_t(outs[i].value+ob.value)==expected[i],"batch evaluator value");
   require(uint32_t(outs[i].mac+ob.mac)==uint32_t(uint64_t(mu)*expected[i]),"batch evaluator MAC");
   require(uint32_t(outs[i].input_mac+ob.input_mac)==uint32_t(uint64_t(mu)*xs[i]),"batch evaluator input MAC");}
 }
 std::cout<<"PASS DCF boundaries/random inputs, NEON, serialization, exact ReLU, paper approximation, input/output MACs, fast PRG, batched DCF/ReLU\n";
 }catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
