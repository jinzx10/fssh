#include "aux.h"

extern const std::complex<double> I(0.0, 1.0);
extern const double PI = acos(-1);
extern const double DELTA = 1e-8;

std::function<double(double)> diff(std::function<double(double)> const& f) {
	return [f] (double const& x) -> double {
		return (-f(x-3*DELTA)/60 + 3.0*f(x-2*DELTA)/20 - 3.0*f(x-DELTA)/4
                +f(x+3*DELTA)/60 - 3.0*f(x+2*DELTA)/20 + 3.0*f(x+DELTA)/4) / DELTA;
	};
}

std::function<std::complex<double>(double)> diff(std::function<std::complex<double>(double)> const& f) {
	return [f] (double const& x) -> std::complex<double> {
		return (-f(x-3*DELTA)/60.0 + 3.0*f(x-2*DELTA)/20.0 - 3.0*f(x-DELTA)/4.0
                +f(x+3*DELTA)/60.0 - 3.0*f(x+2*DELTA)/20.0 + 3.0*f(x+DELTA)/4.0) / DELTA;
	};
}

//void set_max_real_positive(arma::cx_vec& col) {
//	arma::uword idx_max = arma::index_max(col);
//	col /= std::exp( I * std::arg(col(idx_max)) );
//}
//
//arma::cx_mat pure_denmat(arma::uword sz) {
//	arma::mat denmat = arma::zeros(sz, sz);
//	denmat(0,0) = 1.0;
//	return arma::cx_mat(denmat, arma::zeros(sz,sz));
//}
