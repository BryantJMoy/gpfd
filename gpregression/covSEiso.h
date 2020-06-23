/* Squared exponential covariance function, C++ implementation
 * Copyright (C) 2020 JBrandon Duck-Mayr
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef COVSEISO_H
#define COVSEISO_H

#include <RcppArmadillo.h> // OK as I know they use include guards

inline arma::mat covSEiso(const arma::mat& sqdist, // -0.5 * (x1 - x2)^2
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

inline arma::mat covSEiso2(const arma::mat& sqdist, // -0.5 * (x1 - x2)^2
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

