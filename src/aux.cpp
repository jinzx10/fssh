#include "aux.h"

extern const std::complex<double> I(0.0, 1.0);
extern const double PI = acos(-1);
extern const double DELTA = 1e-3;

int pm(bool state) {return state ? 1 : -1;}

template <>  num_t<false> keep_cplx<false>(std::complex<double> const& z) { return z.real(); }

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
