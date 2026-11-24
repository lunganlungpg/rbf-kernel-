# Kernel ridge regression and classification with an RBF kernel.
#
# Fit:      (K + lambda I) alpha = Y - mean(Y)
# Predict:  f(x) = mean(Y) + sum_i alpha_i K(x, x_i)
# Classification codes the classes as +1/-1 columns (one per class, or a
# single column for two classes) and predicts the class with the largest score.

.prep_x <- function(x, scale, center = NULL, sd = NULL) {
  x <- .as_num_matrix(x, "x")
  if (isTRUE(scale)) {
    if (is.null(center)) {
      center <- colMeans(x)
      sd <- apply(x, 2, stats::sd)
      sd[!is.finite(sd) | sd == 0] <- 1
    }
    x <- sweep(sweep(x, 2, center, "-"), 2, sd, "/")
  }
  list(x = x, center = center, sd = sd)
}

.code_y <- function(y) {
  if (is.factor(y) || is.character(y) || is.logical(y)) {
    y <- droplevels(as.factor(y))
    lev <- levels(y)
    if (length(lev) < 2) stop("the response has only one class", call. = FALSE)
    if (length(lev) == 2L) {
      Y <- matrix(ifelse(y == lev[2], 1, -1), ncol = 1)
    } else {
      Y <- sapply(lev, function(l) ifelse(y == l, 1, -1))
    }
    return(list(Y = Y, task = "classification", levels = lev))
  }
  list(Y = matrix(as.numeric(y), ncol = 1), task = "regression", levels = NULL)
}

.decode <- function(S, levels) {
  if (length(levels) == 2L) factor(ifelse(S[, 1] > 0, levels[2], levels[1]), levels = levels)
  else factor(levels[max.col(S, ties.method = "first")], levels = levels)
}

#' Kernel ridge regression and classification with an RBF kernel
#'
#' Solves `(K + lambda I) alpha = y - mean(y)`. The prediction is a weighted sum
#' of Gaussian bumps centred on the training points. A factor (or character or
#' logical) response gives a classifier: each class is coded +1 / -1 and the
#' class with the largest score wins.
#'
#' @param x Numeric matrix or data frame of features.
#' @param y Response: numeric for regression, factor/character/logical for
#'   classification.
#' @param gamma Kernel coefficient, or `"scale"` for [gamma_scale()] or
#'   `"median"` for [gamma_median()] (computed on the scaled features).
#' @param lambda Ridge penalty. Larger values give smoother fits.
#' @param scale Standardise the features (recommended; the kernel uses raw
#'   distances).
#' @return An object of class `"rbf_krr"`.
#' @seealso [krr_tune()] to choose `gamma` and `lambda`, [rbf_workflow()].
#' @examples
#' fit <- krr(iris[, 1:4], iris$Species, gamma = 0.25, lambda = 0.1)
#' fit
#' table(predicted = predict(fit, iris[, 1:4]), actual = iris$Species)
#'
#' reg <- krr(mtcars[, c("wt", "hp")], mtcars$mpg, lambda = 1)
#' predict(reg, data.frame(wt = 3, hp = 150))
#' @export
krr <- function(x, y, gamma = "scale", lambda = 1, scale = TRUE) {
  px <- .prep_x(x, scale)
  if (nrow(px$x) != length(y)) stop("x and y have different numbers of rows", call. = FALSE)
  if (anyNA(y)) stop("y contains missing values", call. = FALSE)
  if (!is.numeric(lambda) || lambda <= 0) stop("lambda must be positive", call. = FALSE)
  if (identical(gamma, "scale")) gamma <- gamma_scale(px$x)
  if (identical(gamma, "median")) gamma <- gamma_median(px$x)
  .check_gamma(gamma)
  cy <- .code_y(y)
  ymean <- colMeans(cy$Y)
  K <- rbf_kernel(px$x, gamma = gamma)
  diag(K) <- diag(K) + lambda
  alpha <- chol2inv(chol(K)) %*% sweep(cy$Y, 2, ymean)
  structure(list(alpha = alpha, x = px$x, center = px$center, sd = px$sd, scale = scale,
                 gamma = gamma, lambda = lambda, ymean = ymean, task = cy$task,
                 levels = cy$levels, features = colnames(.as_num_matrix(x, "x")), n = nrow(px$x)),
            class = "rbf_krr")
}

#' @rdname krr
#' @param object A fitted `"rbf_krr"` model.
#' @param newdata Features for new observations (same columns as `x`).
#' @param type For classifiers: `"class"` or `"score"` (the decision values).
#' @param ... Unused.
#' @export
predict.rbf_krr <- function(object, newdata, type = c("class", "score"), ...) {
  type <- match.arg(type)
  px <- .prep_x(newdata, object$scale, object$center, object$sd)
  S <- sweep(rbf_kernel(px$x, object$x, gamma = object$gamma) %*% object$alpha, 2, object$ymean, "+")
  if (object$task == "regression") return(as.numeric(S))
  if (type == "score") {
    colnames(S) <- if (length(object$levels) == 2L) object$levels[2] else object$levels
    return(S)
  }
  .decode(S, object$levels)
}

#' @rdname krr
#' @export
print.rbf_krr <- function(x, ...) {
  cat(sprintf("RBF kernel ridge %s on %d observations and %d features\n", x$task, x$n, ncol(x$x)))
  cat(sprintf("  gamma = %.4g (sigma = %.4g), lambda = %.4g, features %s\n", x$gamma,
              gamma_to_sigma(x$gamma), x$lambda, if (x$scale) "standardised" else "unscaled"))
  if (!is.null(x$levels)) cat("  classes:", paste(x$levels, collapse = ", "), "\n")
  invisible(x)
}

#' Tune kernel ridge models by exact leave-one-out cross-validation
#'
#' For kernel ridge the leave-one-out residual has a closed form:
#' `e_i = alpha_i / [(K + lambda I)^-1]_ii`. With one eigendecomposition
#' `K = U D U'` per gamma, every lambda costs only matrix products, so the
#' whole grid is evaluated without refitting. The score is accuracy (balanced
#' accuracy if `balanced = TRUE`) for classification and root mean squared
#' error for regression.
#'
#' @inheritParams krr
#' @param gammas Gamma values to try. Default: 9 values from 1/30 to 30 times
#'   `1 / ncol(x)` (the `"scale"` value for standardised data).
#' @param lambdas Lambda values to try. Default: `10^(-4:2)`.
#' @param balanced Use balanced accuracy (mean per-class recall), better for
#'   imbalanced classes.
#' @return An object of class `"krr_tune"` with `grid` (every combination and
#'   its leave-one-out score), `best` (the winning row) and `model` (refitted
#'   with the best values).
#' @examples
#' tuned <- krr_tune(iris[, 1:4], iris$Species)
#' tuned
#' head(tuned$grid)
#' @export
krr_tune <- function(x, y, gammas = NULL, lambdas = 10^(-4:2), scale = TRUE, balanced = FALSE) {
  px <- .prep_x(x, scale)
  X <- px$x
  n <- nrow(X)
  if (n != length(y)) stop("x and y have different numbers of rows", call. = FALSE)
  if (is.null(gammas)) gammas <- (1 / ncol(X)) * 10^seq(-1.5, 1.5, length.out = 9)
  cy <- .code_y(y)
  Yc <- sweep(cy$Y, 2, colMeans(cy$Y))
  D2 <- sq_dist(X)
  rows <- list()
  for (g in gammas) {
    e <- eigen(exp(-g * D2), symmetric = TRUE)
    U <- e$vectors
    d <- pmax(e$values, 0)
    UtY <- crossprod(U, Yc)
    U2 <- U^2
    for (l in lambdas) {
      inv <- 1 / (d + l)
      alpha <- U %*% (inv * UtY)
      hdiag <- as.numeric(U2 %*% inv)          # diag of (K + lambda I)^-1
      loo <- Yc - alpha / hdiag                # leave-one-out predictions (centred)
      score <- if (cy$task == "regression") {
        sqrt(mean((loo - Yc)^2))
      } else {
        pred <- .decode(loo, cy$levels)
        truth <- factor(as.character(y), levels = cy$levels)
        if (balanced) mean(tapply(pred == truth, truth, mean)) else mean(pred == truth)
      }
      rows[[length(rows) + 1L]] <- data.frame(gamma = g, lambda = l, score = score)
    }
  }
  grid <- do.call(rbind, rows)
  metric <- if (cy$task == "regression") "loo_rmse" else if (balanced) "loo_balanced_accuracy" else "loo_accuracy"
  names(grid)[3] <- metric
  # Best score; ties go to the smoothest model (smallest gamma, then largest lambda)
  ord <- if (cy$task == "regression") order(grid[[metric]], grid$gamma, -grid$lambda)
         else order(-grid[[metric]], grid$gamma, -grid$lambda)
  best <- grid[ord[1], ]
  rownames(best) <- NULL
  model <- krr(x, y, gamma = best$gamma, lambda = best$lambda, scale = scale)
  structure(list(grid = grid, best = best, model = model, metric = metric, task = cy$task),
            class = "krr_tune")
}

#' @rdname krr_tune
#' @param ... Unused.
#' @export
print.krr_tune <- function(x, ...) {
  cat(sprintf("Kernel ridge %s tuned by exact leave-one-out over %d (gamma, lambda) pairs\n",
              x$task, nrow(x$grid)))
  cat(sprintf("  best: gamma = %.4g, lambda = %.4g, %s = %.4f\n", x$best$gamma, x$best$lambda,
              x$metric, x$best[[x$metric]]))
  invisible(x)
}

#' @rdname krr_tune
#' @export
plot.krr_tune <- function(x, ...) {
  g <- sort(unique(x$grid$gamma))
  l <- sort(unique(x$grid$lambda))
  Z <- matrix(NA_real_, length(g), length(l))
  for (i in seq_len(nrow(x$grid))) {
    Z[match(x$grid$gamma[i], g), match(x$grid$lambda[i], l)] <- x$grid[[x$metric]][i]
  }
  rev_cols <- x$task == "regression"
  graphics::image(log10(g), log10(l), Z, col = grDevices::hcl.colors(40, "Viridis", rev = rev_cols),
                  xlab = "log10(gamma)", ylab = "log10(lambda)",
                  main = paste("Leave-one-out", sub("loo_", "", x$metric)), ...)
  graphics::contour(log10(g), log10(l), Z, add = TRUE, col = "white", labcex = 0.7)
  graphics::points(log10(x$best$gamma), log10(x$best$lambda), pch = 4, cex = 2, lwd = 3, col = "red")
  invisible(x)
}
