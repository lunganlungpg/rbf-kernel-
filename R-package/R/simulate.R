#' Simulate classic two-dimensional classification problems
#'
#' Two interleaving half circles (`make_moons()`) or two concentric circles
#' (`make_circles()`): standard non-linearly separable test problems for RBF
#' models.
#'
#' @param n Number of points (split evenly between the classes).
#' @param noise Standard deviation of Gaussian noise added to the points.
#' @param factor Ratio of the inner to the outer circle radius.
#' @param seed Random seed.
#' @return A data frame with columns `x1`, `x2` and `class` (a factor).
#' @examples
#' moons <- make_moons(200, noise = 0.2)
#' plot(moons$x1, moons$x2, col = moons$class, pch = 19)
#' @name simulate
NULL

#' @rdname simulate
#' @export
make_moons <- function(n = 200, noise = 0.2, seed = 1) {
  .with_seed(seed, {
    n1 <- n %/% 2
    n2 <- n - n1
    t1 <- seq(0, pi, length.out = n1)
    t2 <- seq(0, pi, length.out = n2)
    x <- c(cos(t1), 1 - cos(t2)) + stats::rnorm(n, 0, noise)
    y <- c(sin(t1), 0.5 - sin(t2)) + stats::rnorm(n, 0, noise)
    data.frame(x1 = x, x2 = y, class = factor(rep(c("A", "B"), c(n1, n2))))
  })
}

#' @rdname simulate
#' @export
make_circles <- function(n = 200, noise = 0.1, factor = 0.5, seed = 1) {
  .with_seed(seed, {
    n1 <- n %/% 2
    n2 <- n - n1
    t1 <- seq(0, 2 * pi, length.out = n1 + 1)[-1]
    t2 <- seq(0, 2 * pi, length.out = n2 + 1)[-1]
    x <- c(cos(t1), factor * cos(t2)) + stats::rnorm(n, 0, noise)
    y <- c(sin(t1), factor * sin(t2)) + stats::rnorm(n, 0, noise)
    data.frame(x1 = x, x2 = y, class = factor(rep(c("outer", "inner"), c(n1, n2)),
                                             levels = c("outer", "inner")))
  })
}
