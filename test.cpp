#include <complex>
#include <type_traits>
#include "aux.h"
#include "TLS.h"

const double A = 0.01, B = 1.6, C = 0.005, D = 1.0, cpl_phase = 0.1;

auto V00 = [](double x) { return (x>0) ? A * (1.0 - exp(-B*x)) : -A * (1 - exp(B*x)); };
auto V11 = [](double x) { return -V00(x);};
auto V01 = [](double x) { return C * exp(-D*x*x); };
auto V01_cplx = [](double x) { return C * exp(-D*x*x) * exp(I*PI*cpl_phase); };

int main() {
	////////////////////////////////////////
	//				squeeze
	////////////////////////////////////////
	std::cout << std::boolalpha;
	std::cout << "   test squeeze start" << std::endl;

	arma::Col<double>::fixed<1> v = {1};
	arma::Col<double>::fixed<2> u = {1,2};
	arma::Col<std::complex<double>>::fixed<1> c = {1};
	arma::Col<std::complex<double>>::fixed<2> z = {1,2};

	auto dv = squeeze<double,1>(v);
	auto du = squeeze<double,2>(u);
	auto dc = squeeze<std::complex<double>,1>(c);
	auto dz = squeeze<std::complex<double>,2>(z);

	std::cout << std::is_same<decltype(dv), double>::value << std::endl;
	std::cout << std::is_same<decltype(du), arma::vec2>::value << std::endl;
	std::cout << std::is_same<decltype(dc), std::complex<double>>::value << std::endl;
	std::cout << std::is_same<decltype(dz), arma::cx_vec2>::value << std::endl;

	std::cout << "   test squeeze end" << std::endl;


	////////////////////////////////////////
	//				keep_cplx
	////////////////////////////////////////
	std::cout << "   test keep_cplx start" << std::endl;

	std::complex<double> z0(1.1,2.2);
	std::cout << std::is_same<decltype(keep_cplx<true>(z0)), std::complex<double>>::value << std::endl;
	std::cout << std::is_same<decltype(keep_cplx<false>(z0)), double>::value << std::endl;

	std::cout << "   test keep_cplx end" << std::endl;


	////////////////////////////////////////
	//				pt
	////////////////////////////////////////
	std::cout << "   test pt start" << std::endl;

	double x0 = 1.1;
	arma::vec3 v0 = {1.1, 2.2, 3.3};

	std::cout << (pt<1>(x0,0,DELTA) == x0+DELTA) << std::endl;
	std::cout << (pt<3>(v0,1,-DELTA)(1) == v0(1)-DELTA) << std::endl;

	std::cout << "   test pt end" << std::endl;


	////////////////////////////////////////
	//				diff
	////////////////////////////////////////
	
	std::cout << "   test diff start" << std::endl;

	double tol = 1e-10; // error tolerance

	auto f = [](double x) {return x*x-3*x;};
	auto df = [](double x) {return 2*x-3;};

	double x1 = 1.2, x2 = -3.5;
	std::cout << (std::abs(op<1,false>::pardiff1(f)(x1,0) - df(x1)) < tol) << std::endl;
	std::cout << (std::abs(op<1,false>::pardiff1(f)(x2,0) - df(x2)) < tol) << std::endl;

	auto g = [](double x) {return std::sin(x) - std::exp(I*PI*x/2.0);};
	auto dg = [](double x) {return std::cos(x) - I * PI / 2.0 * std::exp(I*PI*x/2.0);}; 

	std::cout << (std::abs(op<1,true>::pardiff1(g)(x1,0) - dg(x1)) < tol) << std::endl;
	std::cout << (std::abs(op<1,true>::pardiff1(g)(x2,0) - dg(x2)) < tol) << std::endl;

	arma::vec3 v1 = {1.1, 2.2, -0.5}, v2 = {0, -0.8, 0};
	auto h = [](arma::vec3 x) {return std::pow(x(0)+x(1),2) - x(1) * x(2); };
	auto dh = [](arma::vec3 x) {return arma::vec3{2*(x(0)+x(1)), 2*(x(0)+x(1))-x(2), -x(1)};};

	std::cout << (arma::norm(op<3,false>::diff1(h)(v1) - dh(v1)) < tol) << std::endl;
	std::cout << (arma::norm(op<3,false>::diff1(h)(v2) - dh(v2)) < tol) << std::endl;



//	auto l = [](arma::vec2 y) {return std::exp(I*y(0)*y(1)); };
//	auto dl = op<2,true>::pardiff1(l);
//	auto dl2 = op<2,true>::diff1(l);
//
//	arma::vec3 xt3 = {0.8, 4.9, 3.7};
//	std::cout << ( std::abs( xt3(1)/h(xt3) - dh(xt3,1) ) < DELTA ) << std::endl;
//	std::cout << std::abs( xt3(1)/h(xt3) - dh(xt3,1) ) << std::endl;
//	std::cout << dh2(xt3) << std::endl;
//
//	arma::vec2 xt4 = {0.9, 1.3};
//	std::cout << ( std::abs( I*xt4(0)*l(xt4)  - dl(xt4,1) ) < DELTA ) << std::endl;
//	std::cout << std::abs( I*xt4(0)*l(xt4)  - dl(xt4,1) ) << std::endl;
//	std::cout << dl2(xt4) << std::endl;
//
//	std::cout << "   test diff end" << std::endl;
//
//
//	////////////////////////////////////////
//	//				TLS
//	////////////////////////////////////////
//	std::cout << "   test TLS start" << std::endl;
//
//	TLS<1, false> tls(V00, V11, V01);
//	TLS<1, true> tls_cplx(V00, V11, V01_cplx);
//
//	std::cout << tls.eigvec(1) << std::endl;
//	std::cout << tls.drvcpl(1,1,1,0) << std::endl;
//	std::cout << tls_cplx.eigvec(1) << std::endl;
//	std::cout << tls_cplx.drvcpl(1,0,0,0) << std::endl;
//
//	//std::cout << tls.V01(0.2) << std::endl;
//	//std::cout << tls.V10(0.2) << std::endl;
//
//	//tls.V(0.2).print();
//
//	//tls.eigval(0.2).print();
//
//	//tls.eigvec(0.2).print();
//
//	//auto g = diff(V00);
//	//std::cout << g(0.02) << std::endl;
	return 0;
}
