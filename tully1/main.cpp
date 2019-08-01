#include <iostream>
#include <cmath>
#include <complex>
#include "Tully1.h"
#include "FSSH.h"

extern const double PI = acos(-1);
extern const std::complex<double> I(0.0, 1.0);

int main() {
	Tully1 model(0.01, 1.6, 0.005, 1.0, 0.3);

	FSSH<Tully1> sys(&model, 0, -0, 0.1, 10);

	sys.eigvecs.print();

	return 0;
}
