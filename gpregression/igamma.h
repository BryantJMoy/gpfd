/* Inverse gamma random number generation and probability density functions
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

#ifndef IGAMMA_H
#define IGAMMA_H

#include <Rcpp.h> // OK as I know they use include guards

inline double rinvgamma(double a, double b) {
    return 1.0 / R::rgamma(a, 1.0 / b);
}

// notice log is default
inline double dinvgamma(double x, double a, double b, int logp = 1) {
    return R::dgamma(1.0 / x, a, 1.0 / b, logp);
}

#endif

