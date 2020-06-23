/* Multivariate normal density and random number generating functions
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

#ifndef MVNORM_H
#define MVNORM_H

#include <RcppArmadillo.h> // OK as I know they use include guards
#include "matops.h" // For regularization

// log only, one observation only
inline double dmvnorm(const arma::vec& x,   // observations
                      const arma::vec& mu,  // mean
                      const arma::mat& S) { // variance
    arma::uword  p = S.n_cols;
    arma::vec    z = x - mu;
    arma::mat    U = arma::chol(S);
    double log_det = arma::sum(arma::log(U.diag())); // really log_det/2
    double     tmp = -1.0 * (S.n_cols/2.0) * M_LN_2PI - log_det;
                 U = arma::inv(arma::trimatu(U));
    return tmp - 0.5 * arma::as_scalar(z.t() * U * U.t() * z);
}

// one observation only
inline arma::vec rmvnorm(const arma::vec& mu, const arma::mat& S) {
    arma::uword m = S.n_cols, i;
    arma::vec res(m);
    for ( i = 0; i < m; ++i ) {
        res[i] = R::rnorm(0.0, 1.0);
    }
    return arma::chol(S, "lower") * res + mu;
}

// Where [x; y] ~ N([mx; my], [Vx, Vyx; Vxy, Vy]), draw from distribution of x|y
inline arma::vec condrmvnorm(const arma::vec& mx,  // x mean
                             const arma::vec& my,  // y mean
                             const arma::mat& Vx,  // x variance
                             const arma::mat& Vy,  // y variance
                             const arma::mat& Vxy, // covariance between x & y
                             const arma::mat& Vyx, // covariance between y & x
                             const arma::vec& y) { // conditioning variable
    using arma::chol;
    using arma::solve;
    using arma::trimatl;
    using arma::trimatu;
    arma::mat L  = chol(Vy, "lower");
    arma::vec mu = mx + Vxy * solve(trimatu(L.t()), solve(trimatl(L), y - my));
    arma::mat M  = arma::solve(arma::trimatl(L), Vyx);
    arma::mat S  = Vx - M.t() * M;
    return rmvnorm(mu, S);
}

#endif

