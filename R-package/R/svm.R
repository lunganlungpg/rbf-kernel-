#' Tune an RBF support vector machine by k-fold cross-validation
#'
#' Needs the 'e1071' package. Features are standardised inside each fold
#' (no leakage), and `gamma` and `cost` are searched together on a grid.
#'
#' @param x Numeric matrix or data frame of features.
#' @param y Factor (classification) or numeric (regression) response.
#' @param gammas Gamma values. Default: 7 values from 1/30 to 30 times
#'   `1 / ncol(x)`.
#' @param costs Cost (C) values. Default: `10^(-1:3)`.
#' @param folds Number of cross-validation folds.
#' @param seed Seed for the fold assignment.
#' @return An object of class `"svm_tune"` with `grid`, `best` and `model` (an
#'   `e1071::svm` fitted to all data with the best values; prediction expects
#'   features standardised with `center` and `sd`, which `predict()` handles).
#' @examples
#' if (requireNamespace("e1071", quietly = TRUE)) {
#'   tuned <- svm_tune(iris[, 1:4], iris$Species, folds = 3)
#'   tuned$best
#'   table(predict(tuned, iris[, 1:4]), iris$Species)
#' }
#' @export
svm_tune <- function(x, y, gammas = NULL, costs = 10^(-1:3), folds = 5, seed = 1) {
  if (!requireNamespace("e1071", quietly = TRUE)) {
    stop("svm_tune() needs the 'e1071' package: install.packages('e1071')", call. = FALSE)
  }
  X <- .as_num_matrix(x, "x")
  n <- nrow(X)
  if (is.null(gammas)) gammas <- (1 / ncol(X)) * 10^seq(-1.5, 1.5, length.out = 7)
  classification <- is.factor(y) || is.character(y) || is.logical(y)
  if (classification) y <- droplevels(as.factor(y))
  fold <- .with_seed(seed, if (classification) {
    f <- integer(n)
    for (l in levels(y)) {
      i <- which(y == l)
      f[i] <- sample(rep_len(seq_len(folds), length(i)))
    }
    f
  } else sample(rep_len(seq_len(folds), n)))
  rows <- list()
  for (g in gammas) for (cst in costs) {
    pred <- if (classification) factor(rep(NA, n), levels = levels(y)) else rep(NA_real_, n)
    for (k in seq_len(folds)) {
      tr <- fold != k
      px <- .prep_x(X[tr, , drop = FALSE], TRUE)
      te <- .prep_x(X[!tr, , drop = FALSE], TRUE, px$center, px$sd)$x
      m <- e1071::svm(px$x, y[tr], kernel = "radial", gamma = g, cost = cst, scale = FALSE)
      pred[!tr] <- stats::predict(m, te)
    }
    score <- if (classification) mean(pred == y) else sqrt(mean((pred - y)^2))
    rows[[length(rows) + 1L]] <- data.frame(gamma = g, cost = cst, score = score)
  }
  grid <- do.call(rbind, rows)
  metric <- if (classification) "cv_accuracy" else "cv_rmse"
  names(grid)[3] <- metric
  ord <- if (classification) order(-grid[[metric]], grid$gamma, grid$cost) else
    order(grid[[metric]], grid$gamma, grid$cost)
  best <- grid[ord[1], ]
  rownames(best) <- NULL
  px <- .prep_x(X, TRUE)
  model <- e1071::svm(px$x, y, kernel = "radial", gamma = best$gamma, cost = best$cost, scale = FALSE)
  structure(list(grid = grid, best = best, model = model, center = px$center, sd = px$sd,
                 metric = metric, task = if (classification) "classification" else "regression"),
            class = "svm_tune")
}

#' @rdname svm_tune
#' @param object An `"svm_tune"` object.
#' @param newdata New features.
#' @param ... Unused.
#' @export
predict.svm_tune <- function(object, newdata, ...) {
  stats::predict(object$model, .prep_x(newdata, TRUE, object$center, object$sd)$x)
}
