#ifndef IGAMMA_H
#define IGAMMA_H

#include <Rcpp.h> // OK as I know they use include guards

double rinvgamma(double a, double b) {
    return 1.0 / R::rgamma(a, 1.0 / b);
}

// log is default
double dinvgamma(double x, double a, double b, int logp = 1) {
    return R::dgamma(1.0 / x, a, 1.0 / b, logp);
}

#endif

