test_that("rbf_kernel gives known values and a valid Gram matrix", {
  K <- rbf_kernel(matrix(c(0, 1, 3), ncol = 1), gamma = 0.5)
  expect_equal(K[1, 2], exp(-0.5))
  expect_equal(K[1, 3], exp(-4.5))
  x <- scale(as.matrix(iris[, 1:4]))
  G <- rbf_kernel(x, gamma = 0.3)
  expect_equal(diag(G), rep(1, 150))
  expect_equal(G, t(G))
  expect_true(all(G > 0 & G <= 1))
  expect_gt(min(eigen(G, symmetric = TRUE, only.values = TRUE)$values), -1e-8)
  expect_equal(dim(rbf_kernel(x[1:5, ], x[1:7, ], 0.3)), c(5L, 7L))
})

test_that("squared distances are never negative and match dist()", {
  x <- matrix(c(1e8, 1e8 + 1e-6, 1e8, 1e8), 2)
  expect_true(all(sq_dist(x) >= 0))
  m <- as.matrix(iris[1:20, 1:4])
  expect_equal(sq_dist(m), unname(as.matrix(dist(m))^2), ignore_attr = TRUE, tolerance = 1e-10)
})

test_that("bandwidth heuristics and conversions", {
  x <- scale(as.matrix(iris[, 1:4]))
  expect_equal(gamma_scale(x), 1 / (4 * var(as.numeric(x))))
  expect_equal(gamma_to_sigma(sigma_to_gamma(2.5)), 2.5)
  expect_equal(sigma_to_gamma(1), 0.5)
  g1 <- gamma_median(as.matrix(iris[, 1:4]))
  g10 <- gamma_median(10 * as.matrix(iris[, 1:4]))
  expect_equal(g10, g1 / 100, tolerance = 1e-10)
  expect_gt(gamma_median(matrix(rnorm(3000), ncol = 2), max_n = 200), 0)
})

test_that("bad inputs give clear errors", {
  expect_error(rbf_kernel(matrix(1:4, 2), gamma = -1), "positive")
  expect_error(rbf_kernel(matrix(c(1, NA), 2)), "missing")
  expect_error(rbf_kernel(iris), "non-numeric")
  expect_error(sq_dist(matrix(1:4, 2), matrix(1:3, 1)), "same number of columns")
  expect_error(gamma_scale(matrix(1, 3, 2)), "zero variance")
})
