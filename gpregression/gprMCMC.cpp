/* Gaussian process regression MCMC sampler
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

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "igamma.h"
#include "mvnorm2.h"
#include "covSEiso.h"
#include "matops.h"

// Inverse of standard normal CDF
inline double phi_inv(double p) {
    return R::qnorm(p, 0.0, 1.0, 1, 0);
}

// Convenience function to update proposal standard deviation
inline double update_step_size(double delta, double current, double target) {
    return (delta * phi_inv(target * 0.5)) / phi_inv(current * 0.5);
}

//' Gaussian process regression MCMC sampler
//' 
//' @param y A numeric vector of outcome observations
//' @param X A numeric matrix of predictor variables,
//'     where rows are observations and columns are variables.
//'     If an intercept is desired the first column should consist of ones.
//' @param S An integer vector of length one giving the number of posterior
//'     samples desired.
//' @param f A numeric vector giving the initial f values to use.
//' @param beta A numeric vector giving the initial beta values to use.
//' @param sy2 A numeric vector of length one giving the initial sigma_y^2
//'     value to use.
//' @param sf2 A numeric vector of length one giving the initial sigma_f^2
//'     value to use.
//' @param ell2 A numeric vector of length one giving the initial ell^2
//'     value to use.
//' @param minloops An integer vector of length one giving the minimun number
//'     of tuning loops to use in tuning the ell^2 proposal standard deviation
//' @param maxloops An integer vector of length one giving the maximun number
//'     of tuning loops to use in tuning the ell^2 proposal standard deviation
//' @param ntune An integer vector of length one the number of sampler
//'     iterations to use in each tuning loop when tuning the ell^2 proposal
//'     standard deviation
//' @param target_rate A numeric vector of length one giving the desired
//'     acceptance rates for ell^2 proposals
//' @param rate_tol A numeric vector of length one giving the tolerance
//'     for acceptance rates for ell^2 proposals; proposal tuning ends when
//'     either the maximum number of tuning loops has been expended or when
//'     delta is between target_rate - rate_tol and target_rate + rate_tol,
//'     whichever comes first
//' @param delta A numeric vector of length one giving an initial value for
//'     the ell^2 proposal standard deviation
//' @param verbose A logical vector of length one; if FALSE, progress updates
//'     are given, but only for what percent of the total iterations have
//'     been completed; if TRUE, information about acceptance rates and
//'     delta values are printed after each tuning loop
//' @param b A numeric vector of length ncol(X) giving the prior mean of beta
//' @param B A numeric matrix giving the prior covariance of beta
//' @param a_y A numeric vector of length one giving the shape for the prior
//'     on sigma_y^2
//' @param b_y A numeric vector of length one giving the scale for the prior
//'     on sigma_y^2
//' @param a_f A numeric vector of length one giving the shape for the prior
//'     on sigma_f^2
//' @param b_f A numeric vector of length one giving the scale for the prior
//'     on sigma_f^2
//' @param a_l A numeric vector of length one giving the shape for the prior
//'     on ell^2
//' @param b_l A numeric vector of length one giving the scale for the prior
//'     on ell^2
//'
//' @return A numeric matrix holding the posterior draws;
//'     each column is a posterior draw and each row is a parameter.
//'     The first n rows are f draws, the next ncol(X) rows are beta draws,
//'     and the last three rows are sigma_y^2, sigma_f^2, and ell^2 draws,
//'     respectively.
// [[Rcpp::export]]
arma::mat gprMCMC(const arma::vec& y,                   // outcomes
                  const arma::mat& X,                   // design matrix
                  const arma::uword S,                  // sampling iterations
                  arma::vec& f,                         // f starting value
                  arma::vec& beta,                      // beta starting value
                  double sy2,                           // sy2 starting value
                  double sf2,                           // sf2 starting value
                  double ell2,                          // ell2 starting value
                  const arma::uword minloops,           // min tuning loops
                  const arma::uword maxloops,           // max tuning loops
                  const arma::uword ntune,              // iters per tuning loop
                  double target_rate,                   // target accrate
                  double rate_tol,                      // accrate tolerance
                  double delta,                         // initial ell2 MH sd
                  bool verbose,                         // print tuning info?
                  const arma::vec& b,                   // beta prior mean
                  const arma::mat& B,                   // beta prior covariance
                  const double a_y, const double b_y,   // s_y^2 prior hypers
                  const double a_f, const double b_f,   // s_f^2 prior hypers
                  const double a_l, const double b_l) { // ell^2 prior hypers
    arma::uword n = y.n_elem; // number of observations
    arma::uword p = X.n_cols; // number of predictors
    // Compute some quantities that will not change during sampling
    arma::vec z = arma::zeros<arma::vec>(n); // vec of 0s for prior mean of f
    double a_y_p = a_y + (0.5 * n); // conditional posterior a for s_y^2
    double a_f_p = a_f + (0.5 * n); // conditional posterior a for s_f^2
    arma::mat  BXT = B * X.t();     // B X^T
    arma::mat   XB = X * B;         // X B
    arma::mat XBXT = XB * X.t();    // X B X^T
    arma::vec   Xb = X * b;         // X b
    arma::mat sqdist(n, n);         // the squared distance part of K (/ -2)
    for ( arma::uword j = 0; j < n; ++j ) {
        sqdist(j, j) = -0.5;
        for ( arma::uword i = j+1; i < n; ++i ) {
            double elem = 0.0;
            for ( arma::uword k = 0; k < p; ++k ) {
                double diff = X(i, k) - X(j, k);
                elem += (diff * diff);
            }
            elem *= -0.5;
            sqdist(i, j) = elem;
            sqdist(j, i) = elem;
        }
    }
    // Set up K from initial values
    arma::mat K = covSEiso(sqdist, sf2, ell2);
    // Useful to store X beta & sigma_y^2 I 
    arma::mat Xbeta = X * beta;
    arma::mat sy2I  = sy2 * arma::eye(n, n);
    // This will hold Cholesky decompositions
    arma::mat L(n, n);
    // Create object to store ell2 draws during tuning
    arma::vec tune_samples(ntune);
    // Keep track of progress
    double prog = 0.0;
    double incr = 0.0;
    if ( verbose ) {
        incr = 1000.0 / ntune;
    } else {
        incr = 1000.0 / (maxloops * (double)ntune);
    }
    double accrate = 0.0;
    int acceptances = 0;
    bool stop_tuning = false;
    int loops_with_high_rates = 0; // for detecting problems
    for ( arma::uword i = 0; i < maxloops; ++i ) {
        if ( stop_tuning ) {
            break;
        }
        acceptances = 0;
        for ( arma::uword s = 0; s < ntune; ++s ) {
            // Periodically update progress and check for user interrupt
            if ( ((s+1) % 10) == 0 ) {
                prog += incr;
                if ( verbose ) {
                    Rprintf("\rTuning loop %2i: ", i+1);
                    Rprintf("%6.2f%% complete", prog);
                } else {
                    Rprintf("\rTuning:   %6.2f%% complete", prog);
                }
                Rcpp::checkUserInterrupt();
            }
            // Draw beta and update Xbeta
            beta = condrmvnorm(b, f + Xb, B, XBXT + sy2I, BXT, XB, y);
            Xbeta = X * beta;
            // Draw f
            f = condrmvnorm(z, Xbeta, K, K + sy2I, K, K, y);
            // Draw sigma_f^2 and update K
            sf2 = rinvgamma(a_f_p, b_f + 0.5 * sf2 * vT_Mi_v(K, f));
            K   = covSEiso(sqdist, sf2, ell2);
            // Draw and ell^2 and update K
            double pv = R::rnorm(ell2, delta);
            if ( pv > 0.0 ) { // only think about accepting positive draws
                double cv_prior = dinvgamma(ell2, a_l, b_l);
                double pv_prior = dinvgamma(pv,  a_l, b_l);
                double cv_ll = dmvnorm(f, z, K);
                double pv_ll = dmvnorm(f, z, covSEiso(sqdist, sf2, pv));
                double r = pv_ll - cv_ll;
                r += (pv_prior - cv_prior);
                if ( std::log(R::runif(0.0, 1.0)) < r ) {
                    acceptances += 1;
                    ell2 = pv;
                    K = covSEiso(sqdist, sf2, ell2);
                } 
            }
            tune_samples[s] = ell2;
            // Draw sigma_y^2 and update sy2I
            arma::vec v = y - f - Xbeta;
            sy2 = rinvgamma(a_y_p, b_y + 0.5 * arma::dot(v, v));
            sy2I = sy2 * arma::eye(n, n);
        }
        // update proposal distribution parameter
        accrate = acceptances / (double)ntune;
        if ( accrate > 0.9 ) {
            loops_with_high_rates += 1;
        } else {
            loops_with_high_rates  = 0;
        }
        if ( verbose ) {
            prog = 0.0;
            Rprintf("\rTuning loop %2i: ", i+1);
            Rprintf("%0.2f ell2 accept rate ", accrate);
            Rprintf("with delta = %7.3f\n", delta);
        }
        if ( accrate < (target_rate + rate_tol)
             && accrate > (target_rate - rate_tol)
             && (i + 1) >= minloops
           ) {
            stop_tuning = true;
            continue;
        }
        if ( loops_with_high_rates > 1 & ((maxloops - i) > 1) ) {
            /* While we are still tuning ell2,
             * it's possible to end up in a bad equilibrium
             * where we're accepting too many upward proposals.
             * We have to snap out of that feedback loop if it occurs.
             */
            loops_with_high_rates = 0;
            ell2     = R::runif(0.0, 5.0);
            delta = -2.0 * phi_inv(target_rate * 0.5);
            if ( verbose ) {
                Rprintf("\n    Divergence detected; resetting\n\n");
            }
        } else {
            // we need to protect against dividing by 0 or -Inf here
            double minrate = 1.0 / (double)ntune;
            double maxrate = (ntune - 1) / (double)ntune;
            accrate  = std::min(maxrate, std::max(minrate, accrate));
            delta = update_step_size(delta, accrate, target_rate);
        }
    }
    if ( verbose ) {
        Rprintf("\nUsing ell2 proposal standard deviation %0.3f\n\n", delta);
    } else {
        Rprintf("\rTuning:   100.00%% complete\n");
    }
    // Create object to store draws
    arma::mat samples(n + p + 3, S);
    for ( arma::uword s = 0; s < S; ++s ) {
        // Periodically update progress and check for user interrupt
        if ( ((s+1) % 10) == 0 ) {
            Rprintf("\rSampling: %6.2f%% complete", 100.0 * ((s+1.0)/S));
            Rcpp::checkUserInterrupt();
        }
        // Draw beta and update Xbeta
        beta = condrmvnorm(b, f + Xb, B, XBXT + sy2I, BXT, XB, y);
        Xbeta = X * beta;
        // Draw f
        f = condrmvnorm(z, Xbeta, K, K + sy2I, K, K, y);
        // Draw sigma_f^2 and update K
        sf2 = rinvgamma(a_f_p, b_f + 0.5 * sf2 * vT_Mi_v(K, f));
        K   = covSEiso(sqdist, sf2, ell2);
        // Draw sigma_f^2 and ell^2 and update K
        double pv = R::rnorm(ell2, std::sqrt(delta));
        if ( pv > 0.0 ) {
            double cv_prior = dinvgamma(ell2, a_l, b_l);
            double pv_prior = dinvgamma(pv,  a_l, b_l);
            double cv_ll = dmvnorm(f, z, K);
            double pv_ll = dmvnorm(f, z, covSEiso(sqdist, sf2, pv));
            double r = pv_ll - cv_ll;
            r += (pv_prior - cv_prior);
            if ( std::log(R::runif(0.0, 1.0)) < r ) {
                acceptances += 1;
                ell2 = pv;
                K = covSEiso(sqdist, sf2, ell2);
            } 
        }
        // Draw sigma_y^2 and update sy2I
        arma::vec v = y - f - Xbeta;
        sy2 = rinvgamma(a_y_p, b_y + 0.5 * arma::dot(v, v));
        sy2I = sy2 * arma::eye(n, n);
        // Store draws
        for ( arma::uword i = 0; i < n; ++ i ) {
            samples(i, s) = f[i];
        }
        for ( arma::uword k = 0; k < p; ++ k ) {
            samples(n + k, s) = beta[k];
        }
        samples(n + p, s) = sy2;
        samples(n + p + 1, s) = sf2;
        samples(n + p + 2, s) = ell2;
    }
    Rprintf("\rSampling: 100.00%% complete\n");
    return samples;
}

