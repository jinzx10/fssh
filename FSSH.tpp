#include "FSSH.h"
#include "aux.h"

extern const double PI;
extern const std::complex<double> I;

template <typename Model>
FSSH<Model>::FSSH():
	model(nullptr), init_state(0), init_denmat(), init_x(0.0), init_v(0.0), dt(1.0),
	init_eigvec(), init_eigval(), state(0), x(0.0), v(0.0), eigvec(), eigval()
{}

template <typename Model>
FSSH<Model>::FSSH(
		Model*			model_,
		int				state0_,
		arma::cx_mat	denmat0_,
		double			x0_,
		double			v0_,
		double  		dt_			):
	model(model_), init_state(state0_), init_denmat(denmat0_), init_x(x0_), init_v(v0_),
	dt(dt_), init_eigvec(), init_eigval()
{
	init();
}

template <typename Model>
void FSSH<Model>::init() {
	arma::eig_sym(init_eigval, init_eigvec, model->V(init_x));

	// adjust initial phase
	// set the largest-magnitude number to be real and positive
	init_eigvec.each_col(set_max_real_positive);

	x = init_x;
	v = init_v;
	state = init_state;
	denmat = init_denmat;
	eigvec = init_eigvec;
	eigval = init_eigval;
}

template <typename Model>
void FSSH<Model>::init(Model* model_, int state0_, arma::cx_mat denmat0_, double x0_, double v0_, double dt_) {
	model = model_;
	init_state = state0_;
	init_denmat = denmat0_;
	init_x = x0_;
	init_v = v0_;
	dt = dt_;
	init();
}

template <typename Model>
void FSSH<Model>::propagate() {


}

//template <typename Model>
//std::complex<double> FSSH<Model>::dc(int i, int j) {
//
//}
