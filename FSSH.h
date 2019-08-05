#ifndef __FSSH_H__
#define __FSSH_H__

#include <armadillo>
#include "aux.h"

template<typename Model>
class FSSH
{
	public:
		FSSH();
		FSSH(
				Model*			model_,
				double			x0_,
				double 			v0_,
				double 			dt_,	
				arma::cx_mat	denmat0_ = {},
				int				state0_ = 0		);

		// initialize
		void					init();
		void					init(Model*, double, double, double, arma::cx_mat, int);

		Model*					model;
		double					dt;

		// initial condition
		double					init_x;
		double 					init_v;
		arma::cx_mat			init_denmat;
		int						init_state;
		arma::cx_mat			init_eigvec;
		arma::vec				init_eigval;

		// dynamical quantities
		int						state;
		arma::cx_mat			denmat;
		double					x;
		double					v;
		arma::cx_mat			eigvec;
		arma::vec				eigval;

		void					propagate();
		//void 					prob();
		std::complex<double>	dc(int, int);

	private:

};

#include "FSSH.tpp"

#endif
