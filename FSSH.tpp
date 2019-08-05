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
	model(model_), init_state(state0_), init_x(x0_), init_v(v0_), dt(dt_),
	init_eigvec(), init_eigval()
{
	init();
}

template <typename Model>
void FSSH<Model>::init() {
	arma::eig_sym(init_eigval, init_eigvec, model->V(init_x));

	// adjust initial phase
	// set the largest-magnitude number to be real
	for (arma::uword i = 0; i != init_eigvec.n_cols; ++i) {
		arma::uword imax = arma::index_max(arma::abs(init_eigvec.col(i)));
		init_eigvec.col(i) /= std::exp( I * std::arg(init_eigvec(imax, i)) );
	}

	x = init_x;
	v = init_v;
	state = init_state;
	eigvec = init_eigvec;
	eigval = init_eigval;
}

//template <typename Model>
//std::complex<double> FSSH<Model>::dc(int i, int j) {
//
//}
