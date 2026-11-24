test_that("classification workflow on real data beats the baseline", {
  wf <- rbf_workflow(breast_cancer, "diagnosis", plot = FALSE)
  expect_s3_class(wf, "rbf_workflow")
  expect_equal(wf$task, "classification")
  acc <- wf$metrics$value[wf$metrics$metric == "accuracy"]
  expect_gt(acc, 0.93)
  expect_gt(acc, wf$baseline$accuracy + 0.2)
  expect_equal(wf$n_train + wf$n_test, 569)
  expect_true(any(grepl("imbalanced", wf$notes)))
  expect_output(print(wf), "Confusion matrix")
  expect_equal(levels(predict(wf, breast_cancer[1:4, ])), c("benign", "malignant"))
})

test_that("stratified split keeps class proportions", {
  wf <- rbf_workflow(iris, "Species", plot = FALSE)
  expect_equal(as.numeric(table(wf$test$actual)), c(12, 12, 12))
})

test_that("regression workflow with a categorical feature and missing values", {
  cars <- transform(mtcars[, c("mpg", "wt", "hp", "cyl")], cyl = factor(cyl))
  wf <- rbf_workflow(cars, "mpg", plot = FALSE)
  expect_equal(wf$task, "regression")
  expect_true(all(c("cyl6", "cyl8") %in% wf$encoded_features))
  expect_lt(wf$metrics$value[1], wf$baseline$RMSE)
  expect_length(predict(wf, cars[1:3, ]), 3)
  aq <- rbf_workflow(airquality[, 1:4], "Ozone", plot = FALSE)
  expect_true(any(grepl("42 row", aq$notes)))
})

test_that("two-dimensional problems, SVM option, files and plots", {
  out <- file.path(tempdir(), "rbf-workflow-test")
  unlink(out, recursive = TRUE)
  wf <- rbf_workflow(make_moons(300, 0.25), "class", output_dir = out, plot = FALSE)
  expect_gt(wf$metrics$value[1], 0.85)
  expect_true(all(file.exists(wf$files)))
  expect_true(all(c("test_metrics.csv", "tuning_grid.csv", "test_predictions.csv",
                    "workflow_plots.png", "report.txt", "per_class_metrics.csv",
                    "confusion_matrix.csv") %in% basename(wf$files)))
  expect_gte(rbf_workflow(make_circles(300, 0.08), "class", plot = FALSE)$metrics$value[1], 0.97)
  skip_if_not_installed("e1071")
  sv <- rbf_workflow(breast_cancer, "diagnosis", method = "svm", plot = FALSE)
  expect_gt(sv$metrics$value[1], 0.93)
  pdf(NULL)
  on.exit(dev.off())
  plot(sv)
  plot(rbf_workflow(make_moons(150, 0.2), "class", method = "svm", plot = FALSE))
})

test_that("workflow input checks", {
  expect_error(rbf_workflow(iris, "nope"), "not found")
  expect_error(rbf_workflow(iris, "Species", features = "nope"), "not found")
  w <- rbf_workflow(transform(iris, grp = as.numeric(Species))[, c(1:4, 6)], "grp", plot = FALSE)
  expect_true(any(grepl("factor\\(\\)", w$notes)))
  expect_error(predict(rbf_workflow(iris, "Species", plot = FALSE), iris[, 1:2]), "missing columns")
})

test_that("simulators give balanced two-class data", {
  m <- make_moons(101)
  expect_equal(nrow(m), 101)
  expect_equal(as.numeric(table(m$class)), c(50, 51))
  expect_identical(make_circles(50, seed = 2), make_circles(50, seed = 2))
  pdf(NULL)
  on.exit(dev.off())
  fit <- krr(m[, 1:2], m$class, gamma = 1, lambda = 0.1)
  expect_s3_class(plot_decision_boundary(fit, m$x1, m$x2, m$class), "factor")
})
