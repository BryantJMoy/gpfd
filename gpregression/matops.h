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

/* The notation here follows Eq. A.5 & A.6 & Alg. 2.1 in R&W (2006).
 * If x and y are jointly normal,
 * A is the covariance of x, B is the covariance of y,
 * and C is the covariance between x and y.
 * mx is the mean of x, my is the mean of y, and ymmy is y minus my.
 * L is the lower Cholesky of A + B.
 */
arma::vec cond_mean(const arma::vec& ymmy, const arma::vec& mx,
                    const arma::mat& C, const arma::mat& L) {
    return mx + C * arma::solve(arma::trimatu(L.t()),
                                arma::solve(arma::trimatl(L), ymmy));
}

arma::mat cond_var(const arma::mat& A, const arma::mat& C, const arma::mat& L) {
    arma::mat M = arma::solve(arma::trimatl(L), C);
    return A - M.t() * M;
}

#endif

