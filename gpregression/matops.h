/* Matrix algebra convenience functions
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

#ifndef MATOPS_H
#define MATOPS_H

#include <RcppArmadillo.h> // OK as I know they use include guards

// regularize (square matrices only)
inline arma::mat reg(const arma::mat& M, const double r = 0.000001) {
    return M + (r * arma::eye(M.n_cols, M.n_cols));
}

inline double vT_Mi_v(const arma::mat & M, const arma::vec& v) {
    using arma::solve;
    using arma::as_scalar;
    using arma::trimatu;
    using arma::trimatl;
    arma::mat U = arma::chol(M);
    return as_scalar(v.t() * solve(trimatu(U), solve(trimatl(U.t()), v)));
}

// This is faster, but might be (?) less stable
// v^T M^(-1) v (for symmetric, posdef M only)
// inline double vT_Mi_v(const arma::mat & M, const arma::vec& v) {
//     return arma::as_scalar(v.t() * arma::inv_sympd(M) * v);
// }

#endif

