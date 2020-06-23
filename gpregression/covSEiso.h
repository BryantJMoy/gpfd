#ifndef COVSEISO_H
#define COVSEISO_H

#include <RcppArmadillo.h> // OK as I know they use include guards

arma::mat covSEiso(const arma::mat& sqdist, // -0.5 * (x1 - x2)^2
                   const double sf2, const double ell2) {
    arma::uword n = sqdist.n_rows;
    arma::mat res(n, n);
    for ( arma::uword i = 0; i < n; ++i ) {
        res(i, i) = sf2 + (0.00000001 * sf2); // regularize
    }
    for ( arma::uword i = 1; i < n; ++i ) {
        for ( arma::uword j = 0; j < i; ++j ) {
            double elem = sf2 * std::exp(ell2 * sqdist(i, j));
            res(i, j) = elem;
            res(j, i) = elem;
        }
    }
    return res;
}

arma::mat covSEiso2(const arma::mat& sqdist, // -0.5 * (x1 - x2)^2
                    const double sf2, const double ell2) {
    arma::uword n = sqdist.n_rows;
    arma::uword m = sqdist.n_cols;
    arma::mat res(n, m);
    for ( arma::uword j = 0; j < m; ++ j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double elem = sf2 * std::exp(ell2 * sqdist(i, j));
            res(i, j) = elem;
        }
    }
    return res;
}

#endif

