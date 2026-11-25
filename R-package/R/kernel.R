#' Squared Euclidean distances between rows
#'
#' Uses `||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y` (one matrix product) and
#' clips tiny negative values caused by rounding.
#'
#' @param x Numeric matrix (or data frame) with one row per observation.
#' @param y Optional second matrix with the same number of columns.
#' @return An `nrow(x)` by `nrow(y)` matrix.
#' @examples
#' sq_dist(matrix(1:4, 2))
#' @export
sq_dist <- function(x, y = NULL) {
  x <- .as_num_matrix(x, "x")
  same <- is.null(y)
  y <- if (same) x else .as_num_matrix(y, "y")
  if (ncol(x) != ncol(y)) stop("x and y must have the same number of columns", call. = FALSE)
  d <- outer(rowSums(x^2), rowSums(y^2), "+") - 2 * tcrossprod(x, y)
  d[d < 0] <- 0
  if (same) diag(d) <- 0
  d
}

#' RBF (Gaussian) kernel matrix
#'
#' `K[i, j] = exp(-gamma * ||x_i - y_j||^2)`, with `gamma = 1 / (2 sigma^2)`.
#'
#' @inheritParams sq_dist
#' @param gamma Positive kernel coefficient. Larger values give a narrower
#'   kernel.
#' @return The kernel matrix.
#' @examples
#' K <- rbf_kernel(matrix(c(0, 1, 3), ncol = 1), gamma = 0.5)
#' round(K, 3)
#' @export
rbf_kernel <- function(x, y = NULL, gamma = 1) {
  .check_gamma(gamma)
  exp(-gamma * sq_dist(x, y))
}

#' Bandwidth heuristics and conversions
#'
#' * `gamma_scale()`: `1 / (ncol(x) * var(x))`, scikit-learn's `"scale"`
#'   default. Equals `1 / ncol(x)` for standardised data.
#' * `gamma_median()`: the median heuristic, `sigma` = median distance
#'   between points, so `gamma = 1 / (2 sigma^2)`.
#' * `sigma_to_gamma()`, `gamma_to_sigma()`: convert between the Gaussian
#'   width and gamma.
#'
#' @param x Numeric matrix or data frame.
#' @param max_n Maximum number of rows used by the median heuristic.
#' @param seed Seed for the subsample.
#' @param sigma,gamma Positive numbers.
#' @return A positive number.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' gamma_scale(x)
#' gamma_median(x)
#' gamma_to_sigma(sigma_to_gamma(2))
#' @name gamma-heuristics
NULL

#' @rdname gamma-heuristics
#' @export
gamma_scale <- function(x) {
  x <- .as_num_matrix(x, "x")
  v <- stats::var(as.numeric(x))
  if (!is.finite(v) || v == 0) stop("x has zero variance", call. = FALSE)
  1 / (ncol(x) * v)
}

#' @rdname gamma-heuristics
#' @export
gamma_median <- function(x, max_n = 1000, seed = 1) {
  x <- .as_num_matrix(x, "x")
  if (nrow(x) > max_n) {
    old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv())
    on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()), add = TRUE)
    set.seed(seed)
    x <- x[sample.int(nrow(x), max_n), , drop = FALSE]
  }
  d <- sqrt(sq_dist(x))
  m <- stats::median(d[upper.tri(d)])
  if (m == 0) stop("median distance is zero (too many duplicate rows)", call. = FALSE)
  sigma_to_gamma(m)
}

#' @rdname gamma-heuristics
#' @export
sigma_to_gamma <- function(sigma) {
  if (!is.numeric(sigma) || any(sigma <= 0)) stop("sigma must be positive", call. = FALSE)
  1 / (2 * sigma^2)
}

#' @rdname gamma-heuristics
#' @export
gamma_to_sigma <- function(gamma) {
  .check_gamma(gamma)
  sqrt(1 / (2 * gamma))
}

.check_gamma <- function(gamma) {
  if (!is.numeric(gamma) || length(gamma) != 1L || !is.finite(gamma) || gamma <= 0) {
    stop("gamma must be a single positive number", call. = FALSE)
  }
  invisible(gamma)
}

.as_num_matrix <- function(x, name) {
  if (is.data.frame(x)) {
    ok <- vapply(x, is.numeric, TRUE)
    if (!all(ok)) stop(sprintf("%s has non-numeric columns: %s", name,
                               paste(names(x)[!ok], collapse = ", ")), call. = FALSE)
    x <- as.matrix(x)
  }
  if (is.null(dim(x))) x <- matrix(x, ncol = 1L)
  storage.mode(x) <- "double"
  if (anyNA(x) || any(!is.finite(x))) stop(sprintf("%s contains missing or infinite values", name),
                                          call. = FALSE)
  x
}
