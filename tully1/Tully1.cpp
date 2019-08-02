#include "Tully1.h"

extern const double PI;
extern const std::complex<double> I;

Tully1::Tully1():
	A(0.01),
	B(1.6),
	C(0.005),
	D(1.0),
	cpl_phase(0.0)
{}

Tully1::Tully1(double A_, double B_, double C_, double D_, double cpl_phase_):
	A(A_),
	B(B_),
	C(C_),
	D(D_),
	cpl_phase(cpl_phase_)
{}

double Tully1::V00(double x) {
	return (x>0) ? A * (1.0 - exp(-B*x)) : -A * (1 - exp(B*x));
}

double Tully1::V11(double x) {
	return -V00(x);
}

std::complex<double> Tully1::V01(double x) {
	return C * exp(-D*x*x) * exp(I*PI*cpl_phase);
}

std::complex<double> Tully1::V10(double x) {
	return std::conj(V01(x));
}

arma::cx_mat Tully1::V(double x) {
	return arma::cx_mat{ {V00(x),V01(x)}, {V10(x), V11(x)} };
}

double Tully1::dV00(double x) {
	return (x>0) ? A*B*exp(-B*x) : A*B*exp(B*x);
}

double Tully1::dV11(double x) {
	return -dV00(x);
}

std::complex<double> Tully1::dV01(double x) {
	return -2.0 * D * x * V01(x);
}

std::complex<double> Tully1::dV10(double x) {
	return std::conj(dV01(x));
}

arma::cx_mat Tully1::dV(double x) {
	return arma::cx_mat{ {dV00(x),dV01(x)}, {dV10(x), dV11(x)} };
}

arma::cx_mat Tully1::dV_eig(double x) {
	return eigvec(x).t() * dV(x) * eigvec(x);
}

arma::vec Tully1::eigval(double x) {
	return sort( eig_sym(V(x)) );
}

double Tully1::eigval0(double x) {
	return eigval(x)(0);
}

double Tully1::eigval1(double x) {
	return eigval(x)(1);
}

arma::cx_mat Tully1::eigvec(double x) {
	arma::vec eval;
	arma::cx_mat evec;
	eig_sym(eval, evec, V(x));
	arma::uvec idx_sort = arma::sort_index(eval);
	return evec.cols(idx_sort);
}

arma::vec Tully1::F(double x) {
	return arma::vec{F0(x), F1(x)};
}

double Tully1::F0(double x) {
	return -std::real(dV_eig(x)(0,0));
}

double Tully1::F1(double x) {
	return -std::real(dV_eig(x)(1,1));
}

std::complex<double> Tully1::dc01(double x) {
	return dV_eig(x)(0,1) / (eigval1(x)-eigval0(x));
}

std::complex<double> Tully1::dc10(double x) {
	return -std::conj(dc01(x));
}


