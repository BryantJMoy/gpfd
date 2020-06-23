## Demonstration of the Gaussian process regression MCMC sampler
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

## Compile sampler
Rcpp::sourceCpp("gprMCMC.cpp")
## Source covSEiso function
source("covSEiso.R")
## Source CR plotting function
source("plot_cr.R")
## Set seed for reproducibility
set.seed(138)
## Set sample sizes for simulations
N = c(50
      , 100
      , 250
      , 500
      )
for ( n in N ) {
    sy2   <- abs(rnorm(1))         ## Ensure observation noise is positive
    sf2   <- abs(rnorm(1))         ## Ensure function noise is positive
    ell2  <- abs(rnorm(1, sd = 2)) ## Same w length scale; > range of values
    beta  <- rnorm(2, sd = 2)      ## Coefficients any value
    x    <- rnorm(n)               ## Predictor
    X    <- cbind(1, x)            ## Add 1 for intercept
    K    <- covSEiso(x, x, sqrt(sf2), sqrt(ell2)) ## Kernel
    Kr   <- K+diag(1e-8 * sf2, n)  ## Regularize kernel (for safety)
    f    <- t(mvtnorm::rmvnorm(1, sigma = Kr)) ## True f values
    g    <- f + X %*% beta
    y    <- g + rnorm(n, sd = sqrt(sy2)) ## Outcomes
    plot(x, y, pch = 3, col = "#80808080", main = n) ## View data
    lines(x[order(x)], (X %*% beta)[order(x)])
    lines(x[order(x)], (f + X %*% beta)[order(x)], lty = 2)
    b <- rep(0, ncol(X))           ## Setup hypers
    B <- diag(10, nrow = ncol(X))
    a0 <- 1
    b0 <- 1
    target <- 0.234               ## Target ell^2 proposal acceptance rate
    delta <- qnorm(target/2) * -2 ## Initial proposal sd
    tunetol <- 0.01               ## Tolerance on ell^2 proposal acceptance rate
    iters <- 50000                ## Sampler iterations
    cat("Sampling for n = ", n, "\n\n")
    t1 <- proc.time()
    samples <- gprMCMC(y, X,      ## data
                       iters,     ## sampling iterations
                       rep(0, n), ## staring values; f
                       c(0, 0),   ## beta
                       var(y),    ## sy2
                       1,         ## sf2
                       var(x),    ## ell2
                       4,         ## min tuning loops
                       20,        ## max tuning loops
                       500,       ## iterations per tuning loop
                       target,    ## target acceptance rate
                       tunetol,   ## tolerance on acceptance rate tuning
                       delta,     ## Initial ell2 MH proposal sd
                       TRUE,      ## print tuning info?
                       b, B,      ## beta prior hypers
                       a0, b0,    ## sy2 prior hypers
                       a0, b0,    ## sf2 prior hypers
                       a0, b0)    ## ell2 prior hypers
    t2 <- proc.time()
    timing <- t2 - t1
    cat("\nSampling took", timing["elapsed"], "seconds\n\n")
    ests <- rowMeans(samples[,])#((iters/2)+1):iters])
    low  <- apply(samples[(n+1):nrow(samples), ], 1, quantile, probs = 0.025)
    high <- apply(samples[(n+1):nrow(samples), ], 1, quantile, probs = 0.975)
    comparison <- data.frame(truth = c(beta, sy2, sf2, ell2),
                             estimate = tail(ests, 5),
                             ci = sprintf("[% 6.2f,% 6.2f]", low, high))
    rownames(comparison) <- c("beta0", "beta1", "sy2", "sf2", "ell2")
    print(comparison, digits = 2)
    rates <- unique(coda::rejectionRate(coda::mcmc(t(samples))))
    accrate <- round(1 - setdiff(rates, 0), 3)
    cat("\nell2 accept rate:", accrate, "\n")
    mu  <- function(sample) c(sample[1:n] + X %*% sample[c(n + 1, n + 2)])
    gdraws <- apply(samples, 2, mu)
    gmu <- rowMeans(gdraws)
    ghi <- apply(gdraws, 1, quantile, probs = 0.975)
    glo <- apply(gdraws, 1, quantile, probs = 0.025)
    o   <- order(x)
    opar <- par(mar = c(3, 3, 3, 1) + 0.1)
    plot(x[o], gmu[o], type = "n", ylim = range(g, ghi, glo),
         xlab = "", ylab = "", xaxt = "n", yaxt = "n")
    axis(side = 1, line = -0.75, tick = FALSE)
    axis(side = 2, line = -0.75, tick = FALSE)
    mtext(side = 1, line = 1.75, text = expression(x))
    mtext(side = 2, line = 1.75, text = expression(g))
    polygon(x = c(x[o], rev(x[o])), y = c(ghi[o], rev(glo[o])), border = NA,
            col = "#0087875f")
    lines(x[o], g[o], lty = 2, lwd = 1.5)
    lines(x[o], gmu[o], col = "#008787", lwd = 1.5)
    pos <- "if"(beta[2] > 0, "topleft", "bottomleft")
    legend(pos, bty = "n", pch = c(NA, NA, 15), lty = c(1, 2, NA),
           col = c("#008787", "black", "#0087875f"),
           legend = c("Posterior mean", "True value", "95% CI"))
    par(opar)
    draws <- t(samples[c(n+1, n+2), ])
    opar <- par(mar = c(3, 3, 3, 1) + 0.1)
    plot_cr(draws, estimate_type = 17, truth_type = 16, truth = beta,
            main = bquote("CR for"~beta~"draws"),
            ypad = c(-0.1, 0.2),
            xlab = "", ylab = "", xaxt = "n", yaxt = "n")
    axis(side = 1, line = -0.75, tick = FALSE)
    axis(side = 2, line = -0.75, tick = FALSE)
    mtext(side = 1, line = 1.75, text = expression(beta[0]))
    mtext(side = 2, line = 1.75, text = expression(beta[1]))
    legend("top", bty = "n", horiz = TRUE, pch = 17:15,
           col = c("#008787", "black", "#0087875f"),
           legend = c("Posterior mean", "True value", "95% CR"))
    par(opar)
    draws <- samples[n+3, ]
    opar <- par(mar = c(3, 3, 3, 1) + 0.1)
    plot_cr(draws, truth = sy2, from = 0, to = max(draws),
            main = bquote("Density for"~sigma[y]^2~"draws"),
            xlab = "", ylab = "", xaxt = "n", yaxt = "n")
    axis(side = 1, line = -0.75, tick = FALSE)
    axis(side = 2, line = -0.75, tick = FALSE)
    mtext(side = 1, line = 1.75, text = expression(sigma[y]^2))
    mtext(side = 2, line = 1.75, text = "Density")
    legend("topleft", bty = "n", lty = c(1, 2, NA), pch = c(NA, NA, 15),
          col = c("#008787", "black", "#0087875f"),
          legend = c("Posterior mean", "True value", "95% CI"))
    par(opar)
    draws <- samples[n+4, ]
    opar <- par(mar = c(3, 3, 3, 1) + 0.1)
    plot_cr(draws, truth = sf2, from = 0, to = quantile(draws, probs = 0.99),
            main = bquote("Density for"~sigma[f]^2~"draws"),
            xlab = "", ylab = "", xaxt = "n", yaxt = "n")
    axis(side = 1, line = -0.75, tick = FALSE)
    axis(side = 2, line = -0.75, tick = FALSE)
    mtext(side = 1, line = 1.75, text = expression(sigma[f]^2))
    mtext(side = 2, line = 1.75, text = "Density")
    legend("topright", bty = "n", lty = c(1, 2, NA), pch = c(NA, NA, 15),
           col = c("#008787", "black", "#0087875f"),
           legend = c("Posterior mean", "True value", "95% CI"))
    par(opar)
    draws <- samples[n+5, ]
    opar <- par(mar = c(3, 3, 3, 1) + 0.1)
    plot_cr(draws, truth = ell2, from = 0, to = quantile(draws, probs = 0.99),
            main = bquote("Density for"~lambda^2~"draws"),
            xlab = "", ylab = "", xaxt = "n", yaxt = "n")
    axis(side = 1, line = -0.75, tick = FALSE)
    axis(side = 2, line = -0.75, tick = FALSE)
    mtext(side = 1, line = 1.75, text = expression(lambda^2))
    mtext(side = 2, line = 1.75, text = "Density")
    legend("topright", bty = "n", lty = c(1, 2, NA), pch = c(NA, NA, 15),
           col = c("#008787", "black", "#0087875f"),
           legend = c("Posterior mean", "True value", "95% CI"))
    par(opar)
    ## Uncomment the following if you want to save results for later examination
    # save(x, y, beta, f, sy2, sf2, ell2, timing, samples,
    #      file = paste0("~/gpr-samples-n", n, ".RData"))
    cat("\n\n\n")
}
