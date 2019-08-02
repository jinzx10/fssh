#include <iostream>
#include <cmath>
#include <complex>
#include "Tully1.h"
#include "FSSH.h"

extern const double PI = acos(-1);
extern const std::complex<double> I(0.0, 1.0);

int main() {
	Tully1 model(0.01, 1.6, 0.005, 1.0, 0.3);

	arma::uword nx = 1000;
	double xmin = -10;
	double xmax = 10;
	arma::vec xgrid = arma::linspace(xmin, xmax, nx);

	arma::mat x_eigval = arma::zeros(nx, 3);
	arma::cx_mat x_dc = arma::zeros<arma::cx_mat>(nx, 2);

	x_eigval.col(0) = xgrid;
	x_dc.col(0) = arma::cx_vec{xgrid, arma::zeros(nx)};

	arma::uvec idx = {1,2};
	for (arma::uword i = 0; i < nx; ++i) {
		x_eigval.row(i).cols(1,2) = model.eigval(xgrid(i)).t();
		x_dc(i,1) = model.dc01(xgrid(i));
	}

	x_eigval.save("eigval.txt", arma::raw_ascii);
	x_dc.save("dc.txt", arma::raw_ascii);

	//FSSH<Tully1> sys(&model, 0, -0, 0.1, 10);

	//sys.eigvecs.print();

	return 0;
}
