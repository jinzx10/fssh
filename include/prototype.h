#ifndef __PROTOTYPE_H__
#define __PROTOTYPE_H__

#include <armadillo>
#include <type_traits>

namespace fssh {

    /* Prototype class for fewest-switches surface hopping */
    template <size_t dim_nucl_crd = 1, bool is_cplx = false>
    struct Prototype
    {
        using NuclCrd = typename std::conditional< dim_nucl_crd == 1, double, arma::Col<double>::fixed<dim_nucl_crd> >::type;
        using ElecMat = typename std::conditional< is_cplx, arma::cx_mat, arma::mat >::type;

        virtual arma::uword     sz_elec() = 0;
        virtual arma::vec       energies(NuclCrd const& x) = 0;
        virtual double          energy(NuclCrd const& x, arma::uword const& state) = 0;
        virtual NuclCrd         force(NuclCrd const& x, arma::uword const& state) = 0;
        virtual ElecMat         drvcpl(NuclCrd const& x) = 0;
    };

}

#endif
