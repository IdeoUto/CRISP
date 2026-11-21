#include "kernels.h"
#include <set>
#include <cmath>
#include <stdexcept>
#include <omp.h>
namespace crisp {
size_t pow2(size_t n){size_t r=1;while(r<n)r*=2;return r;}
seal::EncryptionParameters Parameters::make()const{if(degree!=8192&&degree!=16384)throw std::invalid_argument("degree must be 8192 or 16384");seal::EncryptionParameters p(seal::scheme_type::ckks);p.set_poly_modulus_degree(degree);p.set_coeff_modulus(seal::CoeffModulus::Create(degree,bits));return p;}
FCPlan::FCPlan(size_t m,size_t n,const std::vector<double>&w,seal::CKKSEncoder&e,double scale):rows(m),cols(n),slots(e.slot_count()),L(pow2(n)),P(slots/L),D(0){
 if(!m||!n||L>slots||w.size()!=m*n)throw std::invalid_argument("FC dimensions");D=(pow2(m)+P-1)/P;if(D>L)throw std::invalid_argument("FC output exceeds single ciphertext capacity");
 std::vector<std::vector<double>>packed(D,std::vector<double>(slots)),diag(D,std::vector<double>(slots));
 for(size_t r=0;r<rows;++r)for(size_t c=0;c<cols;++c)packed[r%D][c+L*(r/D)]=w[r*cols+c];
 for(size_t ct=0;ct<D;++ct)for(size_t c=0;c<slots;++c){size_t d=(c-ct)&(D-1);size_t dest=((c-d)&(L-1))+L*(c/L);diag[d][dest]=packed[ct][c];}
 diagonals.resize(D);active.resize(D);
 #pragma omp parallel for schedule(static)
 for(long d=0;d<long(D);++d){active[d]=std::any_of(diag[d].begin(),diag[d].end(),[](double v){return v!=0;});if(active[d])e.encode(diag[d],scale,diagonals[d]);}
}
std::vector<C> FCPlan::pack(const std::vector<C>&x)const{if(x.size()!=cols)throw std::invalid_argument("FC input");std::vector<C>v(slots);for(size_t b=0;b<P;++b)std::copy(x.begin(),x.end(),v.begin()+b*L);return v;}
std::vector<C> FCPlan::unpack(const std::vector<C>&x)const{std::vector<C>v(rows);for(size_t r=0;r<rows;++r)v[r]=x[r%D+L*(r/D)];return v;}
std::vector<int> FCPlan::rotations()const{std::vector<int>r;for(size_t i=1;i<D;++i)r.push_back(i);for(size_t i=D;i<L;i*=2)r.push_back(i);return r;}
seal::Ciphertext FCPlan::eval(const seal::Ciphertext&x,seal::Evaluator&e,const seal::GaloisKeys&g,seal::Encryptor&en)const{seal::Ciphertext out;bool initialized=false;for(size_t i=0;i<D;++i){if(!active[i])continue;seal::Ciphertext t;if(i)e.rotate_vector(x,i,g,t);else t=x;e.multiply_plain_inplace(t,diagonals[i]);if(!initialized){out=std::move(t);initialized=true;}else e.add_inplace(out,t);}if(!initialized){en.encrypt_zero(x.parms_id(),out);out.scale()=x.scale()*x.scale();}for(size_t i=D;i<L;i*=2){seal::Ciphertext t;e.rotate_vector(out,i,g,t);e.add_inplace(out,t);}e.rescale_to_next_inplace(out);return out;}
ConvPlan::ConvPlan(ConvShape s,const std::vector<double>&w,seal::CKKSEncoder&e,double sc):shape(s),slots(e.slot_count()),L(pow2(s.h*s.w)),P(slots/L),inputs(0),outputs(0),scale(sc),weights(w){if(!s.h||!s.w||!s.ci||!s.co||!s.k||s.stride!=1||L>slots||w.size()!=s.ci*s.co*s.k*s.k)throw std::invalid_argument("conv dimensions/stride");inputs=(s.ci+P-1)/P;outputs=(s.co+P-1)/P;}
std::vector<std::vector<C>>ConvPlan::pack(const std::vector<C>&x)const{if(x.size()!=shape.ci*shape.h*shape.w)throw std::invalid_argument("conv input");std::vector<std::vector<C>>v(inputs,std::vector<C>(slots));for(size_t c=0;c<shape.ci;++c)std::copy_n(x.begin()+c*shape.h*shape.w,shape.h*shape.w,v[c/P].begin()+(c%P)*L);return v;}
std::vector<C>ConvPlan::unpack(const std::vector<std::vector<C>>&x)const{std::vector<C>y(shape.co*shape.oh()*shape.ow());for(size_t c=0;c<shape.co;++c)for(size_t r=0;r<shape.oh();++r)for(size_t s=0;s<shape.ow();++s)y[(c*shape.oh()+r)*shape.ow()+s]=x[c/P][(c%P)*L+r*shape.w+s];return y;}
std::vector<int>ConvPlan::rotations()const{std::set<int>r;for(size_t d=0;d<P;++d)for(size_t ky=0;ky<shape.k;++ky)for(size_t kx=0;kx<shape.k;++kx){bool active=false;for(size_t lane=0;lane<std::min(P,shape.co);++lane)if((lane+d)%P<std::min(P,shape.ci))active=true;if(!active)continue;int shift=int(d*L)+int(ky*shape.w+kx)-int(shape.pad*(shape.w+1));shift=(shift%int(slots)+int(slots))%int(slots);if(shift)r.insert(shift);}return {r.begin(),r.end()};}
std::vector<seal::Ciphertext>ConvPlan::eval(const std::vector<seal::Ciphertext>&x,seal::Evaluator&e,seal::CKKSEncoder&enc,const seal::GaloisKeys&g,int threads,seal::Encryptor&en)const{
 if(x.size()!=inputs)throw std::invalid_argument("conv ciphertext count");std::vector<seal::Ciphertext>out(outputs);std::vector<uint8_t>initialized(outputs);
 // Cache only one input group's rotations at a time; reuse across all output
 // groups. Memory is bounded by P*k*k rather than ci*co*k*k ciphertexts.
 for(size_t ic=0;ic<inputs;++ic){std::map<int,seal::Ciphertext>rot;rot.emplace(0,x[ic]);for(int step:rotations()){seal::Ciphertext t;e.rotate_vector(x[ic],step,g,t);rot.emplace(step,std::move(t));}
 #pragma omp parallel for num_threads(threads) schedule(static)
 for(long oc=0;oc<long(outputs);++oc){std::vector<double>maskv(slots);seal::Plaintext pt;
 for(size_t d=0;d<P;++d)for(size_t ky=0;ky<shape.k;++ky)for(size_t kx=0;kx<shape.k;++kx){std::fill(maskv.begin(),maskv.end(),0);bool any=false;
 for(size_t lane=0;lane<P;++lane){size_t outc=size_t(oc)*P+lane,inc=ic*P+(lane+d)%P;if(outc>=shape.co||inc>=shape.ci)continue;double w=weights[((outc*shape.ci+inc)*shape.k+ky)*shape.k+kx];if(w==0)continue;
 for(size_t r=0;r<shape.oh();++r)for(size_t c=0;c<shape.ow();++c){int sy=int(r+ky)-int(shape.pad),sx=int(c+kx)-int(shape.pad);if(sy>=0&&sx>=0&&sy<int(shape.h)&&sx<int(shape.w)){maskv[lane*L+r*shape.w+c]=w;any=true;}}}
 if(!any)continue;int step=int(d*L+ky*shape.w+kx)-int(shape.pad*(shape.w+1));step=(step%int(slots)+int(slots))%int(slots);enc.encode(maskv,scale,pt);seal::Ciphertext term;e.multiply_plain(rot.at(step),pt,term);if(!initialized[oc]){out[oc]=std::move(term);initialized[oc]=1;}else e.add_inplace(out[oc],term);
 }} }
 for(size_t oc=0;oc<outputs;++oc){if(!initialized[oc]){en.encrypt_zero(x[0].parms_id(),out[oc]);out[oc].scale()=x[0].scale()*scale;}e.rescale_to_next_inplace(out[oc]);}return out;
}
std::vector<C>plain_fc(size_t m,size_t n,const std::vector<double>&w,const std::vector<C>&x){std::vector<C>y(m);for(size_t r=0;r<m;++r)for(size_t c=0;c<n;++c)y[r]+=w[r*n+c]*x[c];return y;}
std::vector<C>plain_conv(ConvShape s,const std::vector<double>&w,const std::vector<C>&x){std::vector<C>y(s.co*s.oh()*s.ow());for(size_t o=0;o<s.co;++o)for(size_t r=0;r<s.oh();++r)for(size_t c=0;c<s.ow();++c)for(size_t i=0;i<s.ci;++i)for(size_t ky=0;ky<s.k;++ky)for(size_t kx=0;kx<s.k;++kx){int sy=int(r+ky)-int(s.pad),sx=int(c+kx)-int(s.pad);if(sy>=0&&sx>=0&&sy<int(s.h)&&sx<int(s.w))y[(o*s.oh()+r)*s.ow()+c]+=w[((o*s.ci+i)*s.k+ky)*s.k+kx]*x[(i*s.h+sy)*s.w+sx];}return y;}
}
