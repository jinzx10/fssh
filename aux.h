#ifndef __AUX_H__
#define __AUX_H__

#include <armadillo>
#include <cmath>
#include <complex>

extern const std::complex<double> I;
extern const double PI;

void set_max_real_positive(arma::cx_vec& col);
arma::cx_mat pure_denmat(arma::uword sz);

// this is a special case: function does not explicitly depend on t
template <typename T>
T rk4_step(T yn, double dt, std::function<T(T)> f); 
//T rk4_step(double t, T yn, double dt, std::function<T(double,T)> f);

#include "aux.tpp"

#endif
