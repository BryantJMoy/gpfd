## Squared exponential covariance function, R implementation
## Copyright (C) 2020 JBrandon Duck-Mayr
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

#' Squared exponential covarance function
#' 
#' @param x1 A numeric vector or matrix; input variables
#' @param x2 A numeric vector or matrix; (perhaps another set of)
#'     input variables. The default is to use x1, i.e. K(x1, x1).
#' @param scale_factor A numeric vector of length one, the scale factor
#' @param length_scale A numeric vector of length one, the lenth scale
#' 
#' @return A numeric matrix whose i,j entry gives the covariance between
#'     function outputs at x1_i and at x2_j
covSEiso <- function(x1, x2 = x1, scale_factor = 1, length_scale = 1) {
    if ( !is.matrix(x1) ) {
        x1 <- matrix(x1, ncol = 1)
    }
    if ( !is.matrix(x2) ) {
        x2 <- matrix(x2, ncol = 1)
    }
    sf     <- scale_factor^2
    ell    <- -0.5 * (1 / (length_scale^2))
    n      <- nrow(x1)
    m      <- nrow(x2)
    d      <- ncol(x1)
    result <- matrix(0, nrow = n, ncol = m)
    for ( j in 1:m ) {
        for ( i in 1:n ) {
            result[i, j] <- sf * exp(ell * sum((x1[i, ] - x2[j, ])^2))
        }
    }
    return(result)
}
