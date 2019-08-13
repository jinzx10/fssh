#include "aux.h"

const std::complex<double> I(0.0, 1.0);
const double PI = acos(-1);
const double DELTA = 1e-8;

void set_max_real_positive(arma::cx_vec& col) {
	arma::uword idx_max = arma::index_max(col);
	col /= std::exp( I * std::arg(col(idx_max)) );
}

arma::cx_mat pure_denmat(arma::uword sz) {
	arma::mat denmat = arma::zeros(sz, sz);
	denmat(0,0) = 1.0;
	return arma::cx_mat(denmat, arma::zeros(sz,sz));
}
