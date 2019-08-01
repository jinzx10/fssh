#include "FSSH.h"

extern const double PI;
extern const std::complex<double> I;

template <typename Model>
FSSH<Model>::FSSH(
		Model*	model_,
		int		state0_,
		double	x0_,
		double	v0_,
		double  dt_			):
	model(model_), state0(state0_), x0(x0_), v0(v0_), dt(dt_), eigvec0(), eigval0()
{
	init();
}

template <typename Model>
void FSSH<Model>::init() {
	arma::eig_sym(eigval0, eigvec0, model->V(x0));

	// adjust initial phase
	for (arma::uword i = 0; i != eigvec0.n_cols; ++i) {
		arma::uword imax = arma::index_max(arma::abs(eigvec0.col(i)));
		eigvec0.col(i) /= std::exp( I * std::arg(eigvec0(imax, i)) );
	}

	x = x0;
	v = v0;
	state = state0;
	eigvecs = eigvec0;
	eigvals = eigval0;
}
