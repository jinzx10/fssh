#include "FSSH.h"
#include "aux.h"

extern const double PI;
extern const std::complex<double> I;

template <typename Model>
FSSH<Model>::FSSH():
	model(nullptr), mass(0.0), dt(0.0), max_num_steps(0),
	init_x(0.0), init_v(0.0), init_denmat(), init_state(0), init_eigvec(), init_eigval(),
	x(0.0), v(0.0), denmat(), state(0), eigvec(), eigval(), step_counter(0)
{}


template <typename Model>
FSSH<Model>::FSSH(
		Model*			model_,
		double			x0_,
		double			v0_,
		double  		mass_,
		double  		dt_,
		arma::uword		max_num_steps_,
		arma::cx_mat	denmat0_,
		arma::uword		state0_		):
	model(model_), mass(mass_), dt(dt_), max_num_steps(max_num_steps_),
	init_x(x0_), init_v(v0_), init_denmat(denmat0_), init_state(state0_),
	init_eigvec(), init_eigval()
{
	init();
}


template <typename Model>
void FSSH<Model>::init() {
	if (init_denmat.is_empty()) {
		init_denmat = pure_denmat(model->num_elec_dofs());
	}

	arma::eig_sym(init_eigval, init_eigvec, model->V(init_x));

	// set the largest-magnitude number to be real and positive
	init_eigvec.each_col(set_max_real_positive);

	x = init_x;
	v = init_v;
	denmat = init_denmat;
	state = init_state;
	eigvec = init_eigvec;
	eigval = init_eigval;

	step_counter = 0;
}


template <typename Model>
void FSSH<Model>::reset(Model* model_, double x0_, double v0_, double mass_, double dt_, arma::uword max_num_steps_, arma::cx_mat denmat0_, arma::uword state0_) {
	model = model_;
	mass = mass_;
	dt = dt_;
	max_num_steps = max_num_steps_;
	init_x = x0_;
	init_v = v0_;
	init_denmat = denmat0_;
	init_state = state0_;
	init();
}


template <typename Model>
void FSSH<Model>::run(arma::uword num_trajs) {
	for (arma::uword i = 0; i < num_trajs; ++i) {
		init();
		propagate();
		collect();
	}
}


template <typename Model>
void FSSH<Model>::clear() {
	data_refl = {};
	data_trans = {};
	data_trap = {};
	data_tot = 0;
	init();
}


template <typename Model>
void FSSH<Model>::propagate() {
	while (step_counter < max_num_steps && x < model->xmax && x > model->xmin) {
		// fourth-order Runge-Kutta
		arma::vec xv = {x, v};
		xv = rk4_step(xv, dt, this->rk4_func);
		//arma::vec k1 = dt * rk4_func(xv);
		//arma::vec k2 = dt * rk4_func(xv + 0.5 * k1);
		//arma::vec k3 = dt * rk4_func(xv + 0.5 * k2);
		//arma::vec k4 = dt * rk4_func(xv + k3);
		//xv += (k1 + 2*k2 + 2*k3 + k4) / 6.0;
		x = xv(0);
		v = xv(1);



		step_counter += 1;
	}
}


template <typename Model>
void FSSH<Model>::collect() {
	data_tot += 1;
	if (step_counter == max_num_steps)
		data_trap(state) += 1;
	if (x >= model->xmax)
		data_trans(state) += 1;
	if (x <= model->xmin)
		data_refl(state) += 1;
}

template <typename Model>
double FSSH<Model>::energy() {
	return 0.5*mass*v*v + model->eigval(x)(state);
}


template <typename Model>
arma::vec FSSH<Model>::rk4_func(arma::vec xv) {
	return arma::vec{xv(1), model->F(xv(0))(state)/mass};
}


//template <typename Model>
//std::complex<double> FSSH<Model>::dc(int i, int j) {
//
//}
