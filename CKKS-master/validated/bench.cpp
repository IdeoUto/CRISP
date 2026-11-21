#include "kernels.h"
#include "../../CRISP/bench/net.h"
#include <openssl/rand.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>
#include <omp.h>
using namespace seal;using namespace crisp;
template<class T>void send_obj(Socket&s,const T&v){std::ostringstream o(std::ios::binary);v.save(o);s.blob(o.str());}
template<class T>void recv_obj(Socket&s,T&v,const SEALContext&c){std::istringstream i(s.blob(),std::ios::binary);v.load(c,i);}
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("usage: crisp_linear 0|1 fc|conv A|B|case degree threads port output.json index");
 int party=std::stoi(argv[1]),threads=std::stoi(argv[5]),port=std::stoi(argv[6]),index=std::stoi(argv[8]);std::string kind=argv[2],model=argv[3];bool initial=kind=="conv"&&index==0;omp_set_num_threads(threads);Parameters params;params.degree=std::stoul(argv[4]);Socket sock;double start=now();if(!party)sock.accept_at(port);else sock.connect_to(port);
 SEALContext ctx(params.make(),true,sec_level_type::tc128);if(!ctx.parameters_set())throw std::runtime_error(ctx.parameter_error_message());CKKSEncoder enc(ctx);Evaluator ev(ctx);PublicKey pk;SecretKey sk;GaloisKeys g;
 size_t rows=0,cols=0;ConvShape sh{};std::unique_ptr<FCPlan>fc;std::unique_ptr<ConvPlan>conv;std::vector<double>w;std::vector<C>x;std::vector<int>steps;
 if(kind=="fc"){if(model=="A"){rows=index==0?100:10;cols=index==0?256:100;}else {rows=10;cols=1024;}w.resize(rows*cols);x.resize(cols);}else{
 std::vector<ConvShape>A{{28,28,1,16,5,0},{12,12,16,16,5,0}};
 std::vector<ConvShape>B{{32,32,3,64,3,1},{32,32,64,64,3,1},{16,16,64,64,3,1},{16,16,64,64,3,1},{8,8,64,64,3,1},{8,8,64,64,1,0},{8,8,64,16,1,0}};
 sh=(model=="A"?A:B).at(index);w.resize(sh.ci*sh.co*sh.k*sh.k);x.resize(sh.ci*sh.h*sh.w);}
 // Public deterministic benchmark fixtures. These are NOT trained weights and
 // NOT secret-shared inference. They isolate the encrypted linear arithmetic.
 for(size_t i=0;i<w.size();++i)w[i]=(int(i%7)-3)*.125;
 constexpr double mu=3;for(size_t i=0;i<x.size();++i){double a=(int(i%11)-5)*.125;x[i]=C(a,mu*a);}
 double prep0=now();if(kind=="fc"){fc=std::make_unique<FCPlan>(rows,cols,w,enc,params.scale);steps=fc->rotations();}else{conv=std::make_unique<ConvPlan>(sh,w,enc,params.scale);steps=conv->rotations();}double prep=now()-prep0;
 if(party){KeyGenerator kg(ctx);kg.create_public_key(pk);sk=kg.secret_key();if(steps.size()>24)kg.create_galois_keys(g);else kg.create_galois_keys(steps,g);send_obj(sock,pk);send_obj(sock,g);}else{recv_obj(sock,pk,ctx);recv_obj(sock,g,ctx);}
 Encryptor en(ctx,pk);std::unique_ptr<Decryptor>de;if(party)de=std::make_unique<Decryptor>(ctx,sk);
 double setup=now()-start-prep;uint64_t setup_sent=sock.sent;std::vector<std::vector<C>>packed=fc?std::vector<std::vector<C>>{fc->pack(x)}:conv->pack(x);size_t nc=packed.size(),no=fc?1:conv->outputs;
 // Server mask is fresh per output slot. A separate validation opening follows
 // the timed protocol. Proofs and InitLin MAC setup are not implemented here.
 double mask_start=now();std::vector<std::vector<C>>server_input(nc,std::vector<C>(enc.slot_count()));for(size_t j=0;j<nc;++j)for(size_t i=0;i<enc.slot_count();++i)server_input[j][i]=C(initial?0:(int((i+j)%17)-8)/16.,(int((i+2*j)%31)-15)/16.);
 std::vector<std::vector<C>>tag_masks(initial?0:nc,std::vector<C>(enc.slot_count()));std::vector<std::vector<C>>masks(no,std::vector<C>(enc.slot_count()));if(!party){std::vector<uint32_t>rnd(2*no*enc.slot_count());if(RAND_bytes((uint8_t*)rnd.data(),rnd.size()*4)!=1)throw std::runtime_error("RAND");for(size_t j=0;j<no;++j)for(size_t i=0;i<enc.slot_count();++i)masks[j][i]=C(double(rnd[2*(j*enc.slot_count()+i)])/4096.,double(rnd[2*(j*enc.slot_count()+i)+1])/4096.);}
 if(!party){for(auto&v:tag_masks){std::vector<uint32_t>r(v.size());if(RAND_bytes((uint8_t*)r.data(),r.size()*4)!=1)throw std::runtime_error("RAND");for(size_t i=0;i<v.size();++i)v[i]=double(r[i])/4096.;}}double mask_prep=now()-mask_start;
 sock.sync(!party);uint64_t sent0=sock.sent;start=now();std::vector<Ciphertext>input(nc),output(no),real(nc),imag(nc),tags(initial?0:nc);
 if(party){for(size_t j=0;j<nc;++j){Plaintext pt;std::vector<C>client=packed[j];for(size_t i=0;i<client.size();++i)client[i]-=server_input[j][i];enc.encode(client,params.scale,pt);en.encrypt(pt,input[j]);if(!initial){std::vector<double>re(enc.slot_count()),im(enc.slot_count());for(size_t i=0;i<re.size();++i){re[i]=client[i].real();im[i]=client[i].imag();}enc.encode(re,params.scale,pt);en.encrypt(pt,real[j]);enc.encode(im,params.scale,pt);en.encrypt(pt,imag[j]);send_obj(sock,real[j]);send_obj(sock,imag[j]);}send_obj(sock,input[j]);}for(auto&c:output)recv_obj(sock,c,ctx);for(auto&c:tags)recv_obj(sock,c,ctx);}
 else {for(size_t j=0;j<nc;++j){if(!initial){recv_obj(sock,real[j],ctx);recv_obj(sock,imag[j],ctx);}recv_obj(sock,input[j],ctx);Plaintext share;enc.encode(server_input[j],params.scale,share);ev.add_plain_inplace(input[j],share);if(!initial){std::vector<double>r(enc.slot_count()),im(enc.slot_count());for(size_t i=0;i<r.size();++i){r[i]=server_input[j][i].real();im[i]=server_input[j][i].imag();}enc.encode(r,params.scale,share);ev.add_plain_inplace(real[j],share);enc.encode(im,params.scale,share);ev.add_plain_inplace(imag[j],share);}}if(fc)output[0]=fc->eval(input[0],ev,g,en);else output=conv->eval(input,ev,enc,g,threads,en);
 for(size_t j=0;j<tags.size();++j){Plaintext a,b;enc.encode(mu*mu*mu,params.scale,a);enc.encode(mu*mu,params.scale,b);ev.multiply_plain(real[j],a,tags[j]);Ciphertext temp;ev.multiply_plain(imag[j],b,temp);ev.sub_inplace(tags[j],temp);ev.rescale_to_next_inplace(tags[j]);Plaintext tm;enc.encode(tag_masks[j],tags[j].parms_id(),tags[j].scale(),tm);ev.sub_plain_inplace(tags[j],tm);}
 for(size_t j=0;j<no;++j){Plaintext m;enc.encode(masks[j],output[j].parms_id(),output[j].scale(),m);ev.sub_plain_inplace(output[j],m);send_obj(sock,output[j]);}for(auto&c:tags)send_obj(sock,c);}
 std::vector<std::vector<C>>decoded_tags(tags.size());std::vector<std::vector<C>>decoded(no);double tag_error=0;if(party){Plaintext p;for(size_t j=0;j<no;++j){de->decrypt(output[j],p);enc.decode(p,decoded[j]);}for(size_t j=0;j<tags.size();++j){de->decrypt(tags[j],p);enc.decode(p,decoded_tags[j]);}}
 double online=now()-start;uint64_t online_sent=sock.sent-sent0;
 // Reveal masking shares to this synthetic-data validator AFTER timing. Never
 // use this validation exchange for confidential model inference.
 uint64_t before=sock.sent;double maxerr=0;
 if(!party){for(auto&m:masks)sock.send(m.data(),m.size()*sizeof(C));for(auto&m:tag_masks)sock.send(m.data(),m.size()*sizeof(C));}
 else{for(size_t j=0;j<no;++j){sock.recv(masks[j].data(),masks[j].size()*sizeof(C));for(size_t i=0;i<decoded[j].size();++i)decoded[j][i]+=masks[j][i];}for(size_t j=0;j<tag_masks.size();++j){sock.recv(tag_masks[j].data(),tag_masks[j].size()*sizeof(C));for(size_t i=0;i<tag_masks[j].size();++i)tag_error=std::max(tag_error,std::abs(decoded_tags[j][i]+tag_masks[j][i]));}auto actual=fc?fc->unpack(decoded[0]):conv->unpack(decoded);auto ref=fc?plain_fc(rows,cols,w,x):plain_conv(sh,w,x);for(size_t i=0;i<ref.size();++i)maxerr=std::max(maxerr,std::abs(actual[i]-ref[i]));}
 std::ofstream o(argv[7]);o<<std::setprecision(12)<<"{\"party\":"<<party<<",\"kind\":\""<<kind<<"\",\"model\":\""<<model<<"\",\"layer\":"<<index<<",\"degree\":"<<params.degree<<",\"slots\":"<<enc.slot_count()<<",\"scale_bits\":40,\"coeff_modulus_bits\":[60,40,40,60],\"threads\":"<<threads<<",\"network\":\"loopback\",\"setup_s\":"<<setup<<",\"mask_preparation_s\":"<<mask_prep<<",\"initial_layer\":"<<(initial?"true":"false")<<",\"plan_preparation_s\":"<<prep<<",\"online_s\":"<<online<<",\"setup_sent_bytes\":"<<setup_sent<<",\"online_sent_bytes\":"<<online_sent<<",\"validation_sent_bytes\":"<<sock.sent-before<<",\"max_abs_error\":"<<maxerr<<",\"tag_max_abs_error\":"<<tag_error<<",\"input_ciphertexts\":"<<nc<<",\"output_ciphertexts\":"<<no<<",\"malicious_security_complete\":false,\"rotation_key_policy\":\"direct up to 24, otherwise signed power-of-two basis\",\"serialization\":\"SEAL default compression\",\"scope\":\"linear component arithmetic benchmark; no ZK or secure InitLin\"}\n";
 if(party&&(maxerr>.005||tag_error>.005))return 2;return 0;
 }catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
