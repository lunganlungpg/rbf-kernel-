test_that("kernel ridge regression matches the direct solution", {
  x <- scale(as.matrix(mtcars[, c("wt", "hp", "disp")]))
  y <- mtcars$mpg
  fit <- krr(x, y, gamma = 0.2, lambda = 0.5, scale = FALSE)
  K <- rbf_kernel(x, gamma = 0.2)
  alpha <- solve(K + 0.5 * diag(32), y - mean(y))
  expect_equal(as.numeric(fit$alpha), unname(alpha), tolerance = 1e-10)
  expect_equal(predict(fit, x), as.numeric(K %*% alpha + mean(y)), tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("closed-form leave-one-out equals brute-force refitting", {
  set.seed(3)
  x <- matrix(rnorm(60), 30)
  y <- sin(x[, 1]) + 0.1 * rnorm(30)
  tuned <- krr_tune(x, y, gammas = 0.7, lambdas = 0.3, scale = FALSE)
  loo <- vapply(1:30, function(i) {
    f <- krr(x[-i, , drop = FALSE], y[-i], gamma = 0.7, lambda = 0.3, scale = FALSE)
    predict(f, x[i, , drop = FALSE])
  }, 0)
  # The tuner centres y once on all rows; refit each fold centring on n - 1
  # rows. Compare with the same centring by refitting on centred y.
  yc <- y - mean(y)
  loo_c <- vapply(1:30, function(i) {
    K <- rbf_kernel(x[-i, , drop = FALSE], gamma = 0.7)
    a <- solve(K + 0.3 * diag(29), yc[-i])
    sum(rbf_kernel(x[i, , drop = FALSE], x[-i, , drop = FALSE], 0.7) * a)
  }, 0)
  expect_equal(tuned$grid$loo_rmse, sqrt(mean((loo_c - yc)^2)), tolerance = 1e-10)
  expect_lt(abs(tuned$grid$loo_rmse - sqrt(mean((loo - y)^2))), 0.05)
})

test_that("classification: binary, multiclass, scores and tuning", {
  fit <- krr(iris[, 1:4], iris$Species, gamma = 0.25, lambda = 0.1)
  expect_s3_class(predict(fit, iris[, 1:4]), "factor")
  expect_gt(mean(predict(fit, iris[, 1:4]) == iris$Species), 0.95)
  expect_equal(dim(predict(fit, iris[1:3, 1:4], type = "score")), c(3L, 3L))
  bin <- krr(breast_cancer[, -1], breast_cancer$diagnosis, lambda = 0.1)
  expect_equal(ncol(predict(bin, breast_cancer[1:2, -1], type = "score")), 1L)
  tuned <- krr_tune(iris[, 1:4], iris$Species)
  expect_s3_class(tuned, "krr_tune")
  expect_gt(tuned$best$loo_accuracy, 0.94)
  expect_equal(nrow(tuned$grid), 9 * 7)
  expect_output(print(tuned), "best")
  expect_output(print(fit), "classification")
  pdf(NULL)
  on.exit(dev.off())
  plot(tuned)
})

test_that("character and logical responses become classes; errors are clear", {
  d <- make_moons(100, 0.2)
  expect_s3_class(predict(krr(d[, 1:2], as.character(d$class)), d[, 1:2]), "factor")
  expect_s3_class(predict(krr(d[, 1:2], d$class == "A"), d[, 1:2]), "factor")
  expect_error(krr(d[, 1:2], d$class[-1]), "different numbers")
  expect_error(krr(d[, 1:2], d$class, lambda = 0), "positive")
  expect_error(krr(d[, 1:2], factor(rep("a", 100))), "one class")
})

test_that("kernel approximations converge to the exact kernel", {
  x <- scale(as.matrix(iris[, 1:4]))
  K <- rbf_kernel(x, gamma = 0.25)
  err <- function(Z) max(abs(tcrossprod(Z) - K))
  expect_lt(err(predict(rff_map(x, 0.25, 5000), x)), err(predict(rff_map(x, 0.25, 50), x)))
  expect_lt(err(predict(nystroem_map(x, 0.25, 150), x)), 1e-6)
  expect_error(predict(rff_map(x, 0.25), x[, 1:2]), "wrong number")
})

test_that("RBF SVM tuning (e1071)", {
  skip_if_not_installed("e1071")
  tuned <- svm_tune(iris[, 1:4], iris$Species, folds = 3)
  expect_gt(tuned$best$cv_accuracy, 0.9)
  expect_gt(mean(predict(tuned, iris[, 1:4]) == iris$Species), 0.9)
  reg <- svm_tune(mtcars[, c("wt", "hp")], mtcars$mpg, folds = 3, costs = c(1, 10))
  expect_true(is.numeric(predict(reg, mtcars[, c("wt", "hp")])))
})
