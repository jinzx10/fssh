#ifndef __AUX_H__
#define __AUX_H__

#include <armadillo>
#include <cmath>
#include <complex>

extern const std::complex<double> I;

auto adj_phase = [](arma::cx_vec& cv) {
	arma::uword idx = cv.index_max();
	cv /= std::exp( I * std::arg(cv(idx)) );
};

#endif
