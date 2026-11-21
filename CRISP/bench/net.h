#pragma once
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <exception>
#include <mutex>
class Socket{
 int fd_=-1;
 void options(){int one=1;setsockopt(fd_,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));timeval tv{300,0};setsockopt(fd_,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));setsockopt(fd_,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv));}
public:
 uint64_t sent=0,received=0;
 Socket()=default;Socket(const Socket&)=delete;Socket&operator=(const Socket&)=delete;
 ~Socket(){if(fd_>=0)close(fd_);}
 void connect_to(int port){for(int t=0;t<1000;++t){fd_=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);inet_pton(AF_INET,"127.0.0.1",&a.sin_addr);if(::connect(fd_,(sockaddr*)&a,sizeof(a))==0){options();return;}close(fd_);fd_=-1;std::this_thread::sleep_for(std::chrono::milliseconds(10));}throw std::runtime_error("connect timeout "+std::to_string(port));}
 void accept_at(int port){int s=socket(AF_INET,SOCK_STREAM,0),one=1;setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);if(bind(s,(sockaddr*)&a,sizeof(a))<0||listen(s,1)<0){close(s);throw std::runtime_error("listen "+std::to_string(port)+": "+strerror(errno));}fd_=accept(s,nullptr,nullptr);close(s);if(fd_<0)throw std::runtime_error("accept");options();}
 void send(const void*p,size_t n){auto q=(const uint8_t*)p;size_t left=n;while(left){auto k=::send(fd_,q,left,0);if(k<=0){if(errno==EINTR)continue;throw std::runtime_error("socket send");}q+=k;left-=k;}sent+=n;}
 void recv(void*p,size_t n){auto q=(uint8_t*)p;size_t left=n;while(left){auto k=::recv(fd_,q,left,0);if(k<=0){if(errno==EINTR)continue;throw std::runtime_error("socket recv timeout/closed");}q+=k;left-=k;}received+=n;}
 template<class T>void put(const T&v){send(&v,sizeof(v));}
 template<class T>T get(){T v;recv(&v,sizeof(v));return v;}
 void blob(const std::string&s){put(uint64_t(s.size()));send(s.data(),s.size());}
 std::string blob(){auto n=get<uint64_t>();if(n>(uint64_t(1)<<31))throw std::runtime_error("oversized message");std::string s(n,'\0');recv(s.data(),n);return s;}
 void sync(bool initiator){uint8_t x=1;if(initiator){put(x);x=get<uint8_t>();}else{x=get<uint8_t>();put(x);}}
};
inline double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline void parallel(size_t n,const std::function<void(size_t)>&fn){std::vector<std::thread> ts;std::exception_ptr error;std::mutex m;for(size_t i=0;i<n;++i)ts.emplace_back([&,i]{try{fn(i);}catch(...){std::lock_guard<std::mutex>l(m);if(!error)error=std::current_exception();}});for(auto&t:ts)t.join();if(error)std::rethrow_exception(error);}
