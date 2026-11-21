// Numerical validation bridge, NOT a two-party inference protocol. The owner of
// this test process has the CKKS secret key and both FSS shares. Never report
// this bridge's wall time as secure inference latency.
#include "kernels.h"
#include "../../CRISP/test/relu/relu.h"
#include <omp.h>
#include <string>
#include <algorithm>
using namespace crisp;using namespace seal;
namespace {struct Engine{Parameters params;SEALContext ctx;KeyGenerator kg;PublicKey pk;GaloisKeys g;CKKSEncoder enc;Evaluator ev;std::unique_ptr<Encryptor>en;Decryptor de;
 Engine():ctx(params.make(),true,sec_level_type::tc128),kg(ctx),enc(ctx),ev(ctx),de(ctx,kg.secret_key()){omp_set_num_threads(4);kg.create_public_key(pk);kg.create_galois_keys(g);en=std::make_unique<Encryptor>(ctx,pk);}
};std::string error;Engine&engine(){static Engine e;return e;}}
extern "C" const char* crisp_last_error(){return error.c_str();}
extern "C" int crisp_linear_layer(int kind,int h,int wi,int ci,int co,int k,int pad,const double*weights,const double*bias,const double*input,double*output,double*max_error){try{
 auto&e=engine();std::vector<C>x;std::vector<double>w;std::vector<std::vector<C>>packed;std::unique_ptr<FCPlan>fc;std::unique_ptr<ConvPlan>cv;
 if(kind==0){w.assign(weights,weights+ci*co);x.resize(ci);fc=std::make_unique<FCPlan>(co,ci,w,e.enc,e.params.scale);}else{w.assign(weights,weights+ci*co*k*k);x.resize(ci*h*wi);cv=std::make_unique<ConvPlan>(ConvShape{size_t(h),size_t(wi),size_t(ci),size_t(co),size_t(k),size_t(pad)},w,e.enc,e.params.scale);}
 for(size_t i=0;i<x.size();++i)x[i]=C(input[i],3*input[i]);packed=fc?std::vector<std::vector<C>>{fc->pack(x)}:cv->pack(x);
 std::vector<Ciphertext>ct(packed.size());Plaintext pt;for(size_t j=0;j<ct.size();++j){e.enc.encode(packed[j],e.params.scale,pt);e.en->encrypt(pt,ct[j]);}
 std::vector<Ciphertext>out=fc?std::vector<Ciphertext>{fc->eval(ct[0],e.ev,e.g,*e.en)}:cv->eval(ct,e.ev,e.enc,e.g,4,*e.en);
 std::vector<std::vector<C>>bv(out.size(),std::vector<C>(e.enc.slot_count()));
 if(fc){for(size_t r=0;r<fc->rows;++r)bv[0][r%fc->D+fc->L*(r/fc->D)]=C(bias[r],3*bias[r]);}
 else{for(size_t c=0;c<cv->shape.co;++c)for(size_t r=0;r<cv->shape.oh();++r)for(size_t s=0;s<cv->shape.ow();++s)bv[c/cv->P][(c%cv->P)*cv->L+r*cv->shape.w+s]=C(bias[c],3*bias[c]);}
 for(size_t j=0;j<out.size();++j){Plaintext bp;e.enc.encode(bv[j],out[j].parms_id(),out[j].scale(),bp);e.ev.add_plain_inplace(out[j],bp);}
 std::vector<std::vector<C>>dec(out.size());for(size_t j=0;j<out.size();++j){e.de.decrypt(out[j],pt);e.enc.decode(pt,dec[j]);}
 auto actual=fc?fc->unpack(dec[0]):cv->unpack(dec);auto ref=fc?plain_fc(co,ci,w,x):plain_conv(cv->shape,w,x);for(size_t i=0;i<ref.size();++i){size_t c=fc?i:i/(cv->shape.oh()*cv->shape.ow());ref[i]+=C(bias[c],3*bias[c]);}*max_error=0;for(size_t i=0;i<actual.size();++i){output[i]=actual[i].real();*max_error=std::max(*max_error,std::abs(actual[i]-ref[i]));}return 0;
 }catch(std::exception&e){error=e.what();return 1;}}
extern "C" int crisp_relu_vector(const double*in,double*out,size_t n,int bits,int mode){try{if(bits<0||bits>24)throw std::invalid_argument("fractional bits");double scale=double(uint64_t(1)<<bits);int failed=0;
 #pragma omp parallel for num_threads(4) reduction(+:failed)
 for(long i=0;i<long(n);++i){double scaled=std::round(in[i]*scale);if(scaled<-2147483648.||scaled>2147483647.){failed++;continue;}uint32_t x=uint32_t(int64_t(scaled));auto keys=generate_authenticated_relu(3,mode==1?ReluMode::paper_approx:ReluMode::exact);uint32_t z=x+keys.first.value.r+keys.second.value.r;auto a=evaluate_authenticated_relu(0,keys.first,z),b=evaluate_authenticated_relu(1,keys.second,z);uint32_t y=a.value+b.value;if(uint32_t(a.mac+b.mac)!=uint32_t(uint64_t(3)*y)||uint32_t(a.input_mac+b.input_mac)!=uint32_t(uint64_t(3)*x))failed++;out[i]=double(int32_t(y))/scale;}
 if(failed)throw std::runtime_error("FSS MAC mismatch or signed fixed-point overflow");return 0;}catch(std::exception&e){error=e.what();return 1;}}
