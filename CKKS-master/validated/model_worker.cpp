#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>
extern "C" int crisp_linear_layer(int,int,int,int,int,int,int,const double*,const double*,const double*,double*,double*);
extern "C" int crisp_relu_vector(const double*,double*,size_t,int,int);
extern "C" const char* crisp_last_error();
void read(void*p,size_t n){if(!std::cin.read((char*)p,n))throw std::runtime_error("truncated request");}
void status(int code){int32_t c=code;std::cout.write((char*)&c,4);if(code){auto e=crisp_last_error();uint32_t n=std::strlen(e);std::cout.write((char*)&n,4);std::cout.write(e,n);}}
int main(){try{std::ios::sync_with_stdio(false);while(true){int32_t kind;if(!std::cin.read((char*)&kind,4))break;if(kind==-1)break;
 if(kind==2){uint32_t h[3];read(h,12);std::vector<double>x(h[0]),y(h[0]);read(x.data(),x.size()*8);int rc=crisp_relu_vector(x.data(),y.data(),x.size(),h[1],h[2]);status(rc);if(!rc)std::cout.write((char*)y.data(),y.size()*8);}
 else{int32_t d[6];read(d,24);int h=d[0],wi=d[1],ci=d[2],co=d[3],k=d[4],pad=d[5];if(h<1||wi<1||ci<1||co<1||k<1)throw std::runtime_error("shape");size_t nw=size_t(ci)*co*(kind?k*k:1),nx=size_t(ci)*(kind?h*wi:1),ny=size_t(co)*(kind?(h+2*pad-k+1)*(wi+2*pad-k+1):1);std::vector<double>w(nw),bias(co),x(nx),y(ny);read(w.data(),nw*8);read(bias.data(),co*8);read(x.data(),nx*8);double err;int rc=crisp_linear_layer(kind,h,wi,ci,co,k,pad,w.data(),bias.data(),x.data(),y.data(),&err);status(rc);if(!rc){std::cout.write((char*)&err,8);std::cout.write((char*)y.data(),ny*8);}}
 std::cout.flush();}}catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
