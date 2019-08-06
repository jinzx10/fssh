#include "aux.h"

template <typename T>
T rk4_step(T yn, double dt, std::function<T(T)> f) {
	T k1 = dt * f( yn );
	T k2 = dt * f( yn + 0.5*k1 );
	T k3 = dt * f( yn + 0.5*k2 );
	T k4 = dt * f( yn + k3 );
	return yn + ( k1 + 2.0*k2 + 2.0*k3 + k4 ) / 6.0;
}
