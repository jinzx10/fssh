#ifndef __AUX_H__
#define __AUX_H__

#include <armadillo>
#include <cmath>
#include <complex>

extern const std::complex<double> I;
extern const double PI;

void set_max_real_positive(arma::cx_vec& col);
arma::cx_mat pure_denmat(arma::uword sz);

#endif
