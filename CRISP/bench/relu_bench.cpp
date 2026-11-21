#include "net.h"
#include "relu.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <iomanip>
uint32_t input_at(size_t i){return uint32_t(int(i%2049)-1024);}
static uint32_t rd32(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("usage: crisp_relu dealer|0|1 count threads baseport paper|exact output.json");
 std::string role=argv[1];size_t count=std::stoull(argv[2]),nt=std::stoull(argv[3]);int port=std::stoi(argv[4]);ReluMode mode=std::string(argv[5])=="paper"?ReluMode::paper_approx:ReluMode::exact;
 if(!count||!nt||nt>32)throw std::runtime_error("invalid count/threads");size_t stride=auth_key_bytes(mode)+8;
 auto begin=[&](size_t t){return count*t/nt;};auto end=[&](size_t t){return count*(t+1)/nt;};
 double start=now();
 std::vector<std::unique_ptr<Socket>>a(nt),b(nt);for(size_t t=0;t<nt;++t){a[t]=std::make_unique<Socket>();b[t]=std::make_unique<Socket>();}
 if(role=="dealer"){
 uint32_t mu=random_ring();parallel(nt,[&](size_t t){a[t]->accept_at(port+100+t);b[t]->accept_at(port+200+t);a[t]->put(mu);a[t]->sync(false);b[t]->sync(false);});
 double setup=now()-start;start=now();
 parallel(nt,[&](size_t t){const size_t batch=256,group=64;std::vector<AuthKey> ka,kb;for(size_t lo=begin(t);lo<end(t);lo+=batch){size_t hi=std::min(lo+batch,end(t));std::vector<uint8_t> wa,wb;wa.reserve(stride*(hi-lo));wb.reserve(stride*(hi-lo));for(size_t g=lo;g<hi;g+=group){size_t gh=std::min(g+group,hi);generate_authenticated_relu_batch(mu,mode,gh-g,ka,kb);for(size_t i=g;i<gh;++i){auto&k0=ka[i-g];auto&k1=kb[i-g];uint32_t x=input_at(i),r=k0.value.r+k1.value.r,z=x+r;uint32_t expected=mode==ReluMode::exact?(x<0x80000000u?x:0):(z>=r?x:0);uint32_t share=random_ring();write_auth_key(wa,k0);put32(wa,share);put32(wa,expected);write_auth_key(wb,k1);put32(wb,x-share);put32(wb,expected);}}a[t]->send(wa.data(),wa.size());b[t]->send(wb.data(),wb.size());}});
 double offline=now()-start;uint64_t bytes=0;for(size_t t=0;t<nt;++t)bytes+=a[t]->sent+b[t]->sent;
 std::ofstream o(argv[6]);o<<std::setprecision(12)<<"{\"role\":\"dealer\",\"count\":"<<count<<",\"threads\":"<<nt<<",\"setup_s\":"<<setup<<",\"offline_s\":"<<offline<<",\"key_bytes_per_party_per_relu\":"<<auth_key_bytes(mode)<<",\"key_distribution_bytes\":"<<2*count*auth_key_bytes(mode)<<",\"fixture_bytes\":"<<16*count<<",\"sent_bytes_including_setup\":"<<bytes<<"}\n";return 0;
 }
 int party=std::stoi(role);if(party!=0&&party!=1)throw std::runtime_error("party");uint32_t mu=0;
 parallel(nt,[&](size_t t){a[t]->connect_to(port+(party?200:100)+t);if(!party){uint32_t m=a[t]->get<uint32_t>();if(t==0)mu=m;}if(!party)b[t]->accept_at(port+t);else b[t]->connect_to(port+t);});
 double setup=now()-start;
 std::vector<std::vector<uint8_t>>keys(nt);std::vector<std::vector<uint32_t>>opened(nt),peer(nt),expected(nt);std::vector<std::vector<AuthOutput>>output(nt);
 for(size_t t=0;t<nt;++t){size_t n=end(t)-begin(t);keys[t].resize(n*stride);opened[t].resize(n);peer[t].resize(n);expected[t].resize(n);output[t].resize(n);}
 parallel(nt,[&](size_t t){a[t]->sync(true);});start=now();parallel(nt,[&](size_t t){a[t]->recv(keys[t].data(),keys[t].size());});double offline=now()-start;
 // Parse the input masks outside online timing using the known wire offsets.
 // The full FSS key deserialization still happens inside the online path, so
 // its cost is charged rather than hidden.
 {const size_t kb=auth_key_bytes(mode),roff=size_t(mode)*FSS_KEY_BYTES;
  for(size_t t=0;t<nt;++t)for(size_t j=0;j<opened[t].size();++j){const uint8_t*p=keys[t].data()+j*stride;opened[t][j]=rd32(p+kb)+rd32(p+roff);expected[t][j]=rd32(p+kb+4);}}
 parallel(nt,[&](size_t t){b[t]->sync(!party);});std::vector<uint64_t>bs(nt);for(size_t t=0;t<nt;++t)bs[t]=b[t]->sent;
 start=now();parallel(nt,[&](size_t t){auto n=opened[t].size();if(!party){b[t]->send(opened[t].data(),4*n);b[t]->recv(peer[t].data(),4*n);}else{b[t]->recv(peer[t].data(),4*n);b[t]->send(opened[t].data(),4*n);}
  const size_t EB=16;std::vector<AuthKey> batch(EB);std::vector<AuthOutput> ob(EB);std::vector<uint32_t> zb(EB);
  for(size_t lo=0;lo<n;lo+=EB){size_t m=std::min(EB,n-lo);
   for(size_t j=0;j<m;++j){const uint8_t*p=keys[t].data()+(lo+j)*stride,*e=p+stride;batch[j]=read_auth_key(p,e,mode);}
   for(size_t j=0;j<m;++j)zb[j]=opened[t][lo+j]+peer[t][lo+j];
   evaluate_authenticated_relu_batch(party,batch.data(),zb.data(),m,ob.data());
   for(size_t j=0;j<m;++j)output[t][lo+j]=ob[j];}
 });double online=now()-start;
 uint64_t sent=0;for(size_t t=0;t<nt;++t)sent+=b[t]->sent-bs[t];
 // Validation only: opens outputs AFTER timing. This is not a claimed malicious
 // consistency-check protocol and its traffic is reported separately.
 size_t wrong=0,approx=0;uint64_t validation=0;
 for(size_t t=0;t<nt;++t){auto before=b[t]->sent;size_t n=output[t].size();if(party)b[t]->send(output[t].data(),n*sizeof(AuthOutput));else{std::vector<AuthOutput>v(n);b[t]->recv(v.data(),n*sizeof(AuthOutput));for(size_t j=0;j<n;++j){uint32_t y=output[t][j].value+v[j].value,m=output[t][j].mac+v[j].mac,u=output[t][j].input_mac+v[j].input_mac,x=input_at(begin(t)+j);wrong+=y!=expected[t][j]||m!=uint32_t(uint64_t(mu)*expected[t][j])||u!=uint32_t(uint64_t(mu)*x);approx+=y!=(x<0x80000000u?x:0);}}validation+=b[t]->sent-before;}
 std::ofstream o(argv[6]);o<<std::setprecision(12)<<"{\"role\":"<<party<<",\"mode\":\""<<argv[5]<<"\",\"count\":"<<count<<",\"threads\":"<<nt<<",\"network\":\"loopback\",\"setup_s\":"<<setup<<",\"offline_receive_s\":"<<offline<<",\"online_s\":"<<online<<",\"online_sent_bytes\":"<<sent<<",\"validation_sent_bytes\":"<<validation<<",\"protocol_errors\":"<<wrong<<",\"signed_relu_mismatches\":"<<approx<<",\"malicious_security_complete\":false}\n";
 return wrong?2:0;
 }catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
