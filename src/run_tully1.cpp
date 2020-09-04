#include "fssh1990.h"
#include "tully.h"
#include <mpi.h>
#include <iostream>
#include <iomanip>

using namespace fssh;

int main() {

    int id, nprocs;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    std::vector<double> klist = {3, 3.5, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.5, 6, 6.5, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    size_t n_trajs = 2000;

    size_t n_trajs_local = n_trajs / nprocs;
    int rem = n_trajs % nprocs;
    if (id < rem)
        n_trajs_local += 1;

    size_t r0_local = 0, r0 = 0;
    size_t r1_local = 0, r1 = 0;
    size_t t0_local = 0, t0 = 0;
    size_t t1_local = 0, t1 = 0;

    Tully1 tully1;
    double mass = 2000;
    FSSH1990 fssh1990(&tully1, mass);

    for (size_t ik = 0; ik != klist.size(); ++ik) {
        double k = klist[ik];
        r0_local = r1_local = t0_local = t1_local = r0 = r1 = t0 = t1 = 0;
        for (size_t it = 0; it != n_trajs_local; ++it) {
            fssh1990.run(-9.9, k / mass, 1, 0, 0, 0, 1);
            if (fssh1990.f[0] > 10) {
                if (fssh1990.state)
                    t1_local += 1;
                else
                    t0_local += 1;
            } else {
                if (fssh1990.state)
                    r1_local += 1;
                else
                    r0_local += 1;
            }
        }

        MPI_Reduce(&r0_local, &r0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&r1_local, &r1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t0_local, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&t1_local, &t1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (id == 0) {
            std::cout /*<< "r0 = "*/ << std::setw(6) << (double)r0 / n_trajs 
                /*<< "    r1 = "*/ << std::setw(6) << (double)r1 / n_trajs
                /*<< "    t0 = "*/ << std::setw(6) << (double)t0 / n_trajs
                /*<< "    t1 = "*/ << std::setw(6) << (double)t1 / n_trajs 
                << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}


