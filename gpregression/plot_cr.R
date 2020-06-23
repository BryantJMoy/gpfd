## Plotting function for credible regions/intervals
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

#' Plot estimate, 95% credible region, and, if known, true value
#' 
#' @param draws The posterior draws.
#'     Must be a vector, for one parameter, or two column matrix, for two.
#' @param level A numeric vector of length one; desired confidence level.
#'     Default is 0.95.
#' @param truth A numeric vector (optionally) giving the true value(s)
#'     (for example, when testing a sampler's performance via simulation).
#' @param estimate_type An integer vector of length one giving the line type
#'     (for one parameter) or point type (for two) to draw the posterior mean
#'     with. The default is 1. If NULL, the posterior mean is not drawn.
#' @param truth_type An integer vector of length one giving the line type
#'     (for one parameter) or point type (for two) to draw the true value(s)
#'     with. The default is 2. If NULL, the truth is not drawn.
#' @param estimate_col Color to plot the estimate in. The default is "#008787".
#' @param truth_col Color to plot the truth in. The default is "black".
#' @param cr_shade A color to draw the confidence region in.
#'     The default is "#0087875f".
#' @param line_width For one parameter, line width (lwd) to mark the estimate
#'     and/or true value.
#' @param point_size For two parameters, point size (cex) to mark the estimate
#'     and/or true value
#' @param xpad Amount to pad default x-axis limits; only used for two parameters
#' @param ypad Amount to pad default y-axis limits; only used for two parameters
#' @param from Lower bound for estimating density of draws for one parameter.
#'     If NA, the default, density()'s default settings are used.
#' @param to Upper bound for estimating density of draws for one parameter.
#'     If NA, the default, density()'s default settings are used.
#' @param ... Other arguments passed to plot()
#' 
#' @section Warning
#' The bivariate confidence region assumes normality of the posterior.
#' 
#' @importFrom mixtools ellipse
plot_cr <- function(draws, level = 0.95, truth = NA,
                    estimate_type = 1, truth_type = 2,
                    estimate_col = "#008787", truth_col = "black",
                    cr_shade = "#0087875f",
                    line_width = 1.5, point_size = 1.5,
                    xpad = c(-0.1, 0.1), ypad = c(-0.1, 0.1),
                    from = NA, to = NA, ...) {
    draws_is_matrix <- is.matrix(draws)
    if ( !draws_is_matrix & is.atomic(draws) & is.numeric(draws) ) {
        if ( is.na(from) ) {
            if ( is.na(to) ) {
                dens <- density(draws)
            } else {
                dens <- density(draws, to = to)
            }
        } else if ( is.na(to) ) {
            dens <- density(draws, from = from)
        } else {
            dens <- density(draws, from = from, to = to)
        }
        qs   <- quantile(draws, probs = c((1 - level)/2, (1 + level)/2))
        low  <- max(which(dens$x <= qs[1]))
        high <- min(which(dens$x >= qs[2]))
        plot(dens, ...)
        x    <- dens$x[c(low, low:high, high)]
        y    <- c(0, dens$y[low:high], 0)
        polygon(x = x, y = y, border = NA, col = cr_shade)
        if ( !is.null(estimate_type) ) {
            abline(v = mean(draws), lty = estimate_type, col = estimate_col,
                   lwd = line_width)
        }
        if ( !is.null(truth_type) & isTRUE(!is.na(truth)) ) {
            abline(v = truth, lty = truth_type, col = truth_col,
                   lwd = line_width)
        }
    } else if ( draws_is_matrix & ncol(draws) == 2 ) {
        mu    <- colMeans(draws)
        Sigma <- cov(draws)
        xy    <- mixtools::ellipse(mu, Sigma, draw = FALSE, alpha = 1 - level)
        xlim  <- range(xy[ , 1])
        ylim  <- range(xy[ , 2])
        if ( all(!is.na(truth)) ) {
            xlim <- range(xlim, truth[1])
            ylim <- range(ylim, truth[2])
        }
        plot(draws, type = "n", xlim = xlim  + xpad, ylim = ylim + ypad, ...)
        polygon(xy, border = NA, col = cr_shade)
        if ( !is.null(estimate_type) ) {
            points(mu[1], mu[2], pch = estimate_type, col = estimate_col,
                   cex = point_size)
        }
        if ( !is.null(truth_type) & all(!is.na(truth)) ) {
            points(truth[1], truth[2], pch = truth_type, col = truth_col,
                   cex = point_size)
        }
    } else {
        stop("draws should be a vector or two column matrix.")
    }
}
