#ifndef __FSSH_H__
#define __FSSH_H__

#include <armadillo>

template<typename Model>
class FSSH
{
	public:
		FSSH();
		FSSH(
				Model*			model_,
				double			x0_,
				double 			v0_,
				double			mass_,
				double 			dt_,
				arma::uword		max_num_steps_ = 5000,
				arma::cx_mat	denmat0_ = {},
				arma::uword		state0_ = 0		);


		void					reset(Model*, double, double, double, double, arma::uword = 5000, arma::cx_mat = {}, arma::uword = 0);
		void					run(arma::uword num_trajs);
		void					clear();

		Model*					model;
		double					mass;
		double					dt;
		arma::uword				max_num_steps;

		// initial condition
		double					init_x;
		double 					init_v;
		arma::cx_mat			init_denmat;
		arma::uword				init_state;
		arma::cx_mat			init_eigvec;
		arma::vec				init_eigval;

		// dynamical quantities
		double					x;
		double					v;
		arma::cx_mat			denmat;
		arma::uword				state;
		arma::cx_mat			eigvec;
		arma::vec				eigval;

		double					energy();
		//void 					prob();
		//std::complex<double>	dc(int, int);

		// data collection
		arma::uvec				data_refl;
		arma::uvec				data_trans;
		arma::uvec				data_trap;
		arma::uword				data_tot;

	private:
		arma::uword				step_counter;
		void					init();
		void					propagate(); // propagate one trajectory
		void					collect();

		arma::vec				rk4_func(arma::vec);

};

#include "FSSH.tpp"

#endif
