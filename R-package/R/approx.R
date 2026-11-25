#' Random Fourier features and the Nystroem approximation
#'
#' Exact kernel methods need the n x n kernel matrix. These maps build an
#' explicit, finite feature matrix `Z` with `Z %*% t(Z)` approximately equal to
#' the RBF kernel matrix, so fast linear methods (e.g. [stats::lm()] or
#' ridge regression) can be used on large data.
#'
#' * `rff_map()` (Rahimi and Recht, 2007): `z(x) = sqrt(2/D) cos(W x + b)`
#'   with `W ~ N(0, 2 gamma I)`, `b ~ U(0, 2 pi)`. Error shrinks like
#'   `1/sqrt(D)`.
#' * `nystroem_map()`: uses `m` landmark rows; `z(x) = K(x, L) K(L, L)^(-1/2)`.
#'   Usually more accurate for the same number of components because it
#'   adapts to the data.
#'
#' @param x Numeric matrix (training data; for Nystroem the landmarks are
#'   sampled from it).
#' @param gamma Kernel coefficient.
#' @param n_components Number of random features (`D`) or landmarks (`m`).
#' @param seed Random seed.
#' @return A map object; call `predict(map, newdata)` to get the feature
#'   matrix for any data.
#' @references Rahimi, A. and Recht, B. (2007). Random features for
#'   large-scale kernel machines. *NIPS*.
#' @examples
#' x <- scale(as.matrix(iris[, 1:4]))
#' K <- rbf_kernel(x, gamma = 0.25)
#' Zr <- predict(rff_map(x, 0.25, 2000), x)
#' Zn <- predict(nystroem_map(x, 0.25, 50), x)
#' c(rff = max(abs(tcrossprod(Zr) - K)), nystroem = max(abs(tcrossprod(Zn) - K)))
#' @name kernel-approximations
NULL

.with_seed <- function(seed, expr) {
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv())
  on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()))
  set.seed(seed)
  expr
}

#' @rdname kernel-approximations
#' @export
rff_map <- function(x, gamma, n_components = 500, seed = 1) {
  x <- .as_num_matrix(x, "x")
  .check_gamma(gamma)
  p <- ncol(x)
  w <- .with_seed(seed, list(W = matrix(stats::rnorm(p * n_components, 0, sqrt(2 * gamma)), p),
                             b = stats::runif(n_components, 0, 2 * pi)))
  structure(c(w, list(gamma = gamma, D = n_components, p = p)), class = "rff_map")
}

#' @rdname kernel-approximations
#' @param object A map from `rff_map()` or `nystroem_map()`.
#' @param newdata Numeric matrix with the same columns as `x`.
#' @param ... Unused.
#' @export
predict.rff_map <- function(object, newdata, ...) {
  x <- .as_num_matrix(newdata, "newdata")
  if (ncol(x) != object$p) stop("newdata has the wrong number of columns", call. = FALSE)
  sqrt(2 / object$D) * cos(sweep(x %*% object$W, 2, object$b, "+"))
}

#' @rdname kernel-approximations
#' @export
nystroem_map <- function(x, gamma, n_components = 100, seed = 1) {
  x <- .as_num_matrix(x, "x")
  .check_gamma(gamma)
  m <- min(n_components, nrow(x))
  idx <- .with_seed(seed, sample.int(nrow(x), m))
  L <- x[idx, , drop = FALSE]
  e <- eigen(rbf_kernel(L, gamma = gamma), symmetric = TRUE)
  keep <- e$values > max(e$values) * 1e-12
  norm <- e$vectors[, keep, drop = FALSE] %*% diag(1 / sqrt(e$values[keep]), sum(keep))
  structure(list(landmarks = L, normalization = norm, gamma = gamma, p = ncol(x)),
            class = "nystroem_map")
}

#' @rdname kernel-approximations
#' @export
predict.nystroem_map <- function(object, newdata, ...) {
  x <- .as_num_matrix(newdata, "newdata")
  if (ncol(x) != object$p) stop("newdata has the wrong number of columns", call. = FALSE)
  rbf_kernel(x, object$landmarks, object$gamma) %*% object$normalization
}
