#ifndef MVNORM_H
#define MVNORM_H

#include <RcppArmadillo.h> // OK as I know they use include guards
#include "matops.h" // For regularization

// log only, one observation only
double dmvnorm(const arma::vec& x, const arma::vec& mu, const arma::mat& S) {
    arma::uword  p = S.n_cols;
    arma::vec    z = x - mu;
    arma::mat    U = arma::chol(S);
    double log_det = arma::sum(arma::log(U.diag())); // really log_det/2
    double     tmp = -1.0 * (S.n_cols/2.0) * M_LN_2PI - log_det;
                 U = arma::inv(arma::trimatu(U));
    return tmp - 0.5 * arma::as_scalar(z.t() * U * U.t() * z);
}

// one observation only
arma::vec rmvnorm(const arma::vec& mu, const arma::mat& S) {
    arma::uword m = S.n_cols, i;
    arma::vec res(m);
    for ( i = 0; i < m; ++i ) {
        res[i] = R::rnorm(0.0, 1.0);
    }
    return arma::chol(S, "lower") * res + mu;
}

arma::vec rmvnorm2(const arma::vec& mu, const arma::mat& S) {
    arma::uword m = S.n_cols, i;
    arma::vec res(m);
    for ( i = 0; i < m; ++i ) {
        res[i] = R::rnorm(0.0, 1.0);
    }
    bool able_to_decompose;
    arma::mat L;
    able_to_decompose = arma::chol(L, S, "lower");
    if ( !able_to_decompose ) {
        arma::chol(L, S + 0.001 * arma::eye(m, m), "lower");
    }
    return L * res + mu;
}

// Where [x; y] ~ N([mx; my], [Vx, Vyx; Vxy, Vy]), draw an observation from x|y
arma::vec condrmvnorm(const arma::vec& mx, const arma::vec& my,   // means
                      const arma::mat& Vx, const arma::mat& Vy,   // variances
                      const arma::mat& Vxy, const arma::mat& Vyx, // covariances
                      const arma::vec& y) {             // conditioning variable
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

