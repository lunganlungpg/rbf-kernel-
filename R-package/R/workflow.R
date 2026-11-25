.encode <- function(df, spec = NULL) {
  if (is.null(spec)) {
    df <- droplevels(df)
    is_cat <- vapply(df, function(v) is.factor(v) || is.character(v) || is.logical(v), TRUE)
    df[is_cat] <- lapply(df[is_cat], as.factor)
    tt <- stats::terms(~ ., data = df)
    xlev <- lapply(df[is_cat], levels)
    spec <- list(terms = stats::delete.response(tt), xlev = xlev, cols = names(df))
  } else {
    missing <- setdiff(spec$cols, names(df))
    if (length(missing)) stop("newdata is missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
    df <- df[spec$cols]
    for (v in names(spec$xlev)) df[[v]] <- factor(as.character(df[[v]]), levels = spec$xlev[[v]])
  }
  mf <- stats::model.frame(spec$terms, df, xlev = spec$xlev, na.action = stats::na.pass)
  X <- stats::model.matrix(spec$terms, mf)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  list(X = X, spec = spec)
}

.split <- function(y, test_fraction, classification, seed) {
  n <- length(y)
  .with_seed(seed, {
    if (classification) {
      test <- unlist(lapply(split(seq_len(n), y), function(i) {
        if (length(i) < 2) integer() else i[sample.int(length(i), max(1, round(length(i) * test_fraction)))]
      }))
    } else {
      test <- sample.int(n, max(1, round(n * test_fraction)))
    }
    seq_len(n) %in% test
  })
}

.metrics <- function(truth, pred, classification) {
  if (classification) {
    cm <- table(predicted = pred, actual = truth)
    recall <- diag(cm) / colSums(cm)
    precision <- diag(cm) / pmax(rowSums(cm), 1)
    f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    list(summary = data.frame(metric = c("accuracy", "balanced_accuracy", "macro_F1"),
                              value = c(mean(pred == truth), mean(recall), mean(f1))),
         per_class = data.frame(class = colnames(cm), precision = as.numeric(precision),
                                recall = as.numeric(recall), F1 = as.numeric(f1),
                                support = as.numeric(colSums(cm))),
         confusion = cm)
  } else {
    r <- truth - pred
    list(summary = data.frame(metric = c("RMSE", "MAE", "R2"),
                              value = c(sqrt(mean(r^2)), mean(abs(r)),
                                        1 - sum(r^2) / sum((truth - mean(truth))^2))))
  }
}

#' Complete RBF kernel workflow for any dataset
#'
#' One call that takes a data frame and the name of the column to predict, then:
#'
#' 1. uses every other column as a feature (categorical columns are one-hot
#'    encoded) and drops rows with missing values;
#' 2. decides between classification (factor, character or logical target) and
#'    regression (numeric target);
#' 3. makes a stratified train/test split;
#' 4. standardises the features using the training rows only;
#' 5. tunes `gamma` and `lambda` of a kernel ridge model by exact leave-one-out
#'    (or `gamma` and `cost` of an RBF SVM by k-fold cross-validation);
#' 6. evaluates the tuned model on the untouched test rows and compares it with
#'    a trivial baseline;
#' 7. flags common problems, and optionally writes tables, a figure and a
#'    report to `output_dir`.
#'
#' @param data A data frame.
#' @param target Name of the column to predict.
#' @param features Feature columns. Default: all other columns.
#' @param method `"krr"` (kernel ridge, no extra packages) or `"svm"` (needs
#'   'e1071').
#' @param test_fraction Fraction of rows held out for testing.
#' @param gammas,lambdas,costs Tuning grids (see [krr_tune()], [svm_tune()]).
#' @param balanced Tune for balanced accuracy. Default: `TRUE` when the largest
#'   class is more than 1.5 times the smallest.
#' @param seed Random seed for the split.
#' @param output_dir Directory to write results to, or `NULL`.
#' @param plot Draw the summary figure.
#' @return An object of class `"rbf_workflow"`. Use `predict(wf, newdata)` to
#'   apply the tuned model (with the same encoding and scaling) to new rows.
#' @examples
#' wf <- rbf_workflow(breast_cancer, target = "diagnosis", plot = FALSE)
#' wf
#'
#' wf2 <- rbf_workflow(make_moons(300, noise = 0.25), "class", plot = FALSE)
#' wf2$metrics
#'
#' # Regression, with a categorical feature
#' cars <- transform(mtcars[, c("mpg", "wt", "hp", "cyl")], cyl = factor(cyl))
#' rbf_workflow(cars, "mpg", plot = FALSE)$metrics
#' @export
rbf_workflow <- function(data, target, features = NULL, method = c("krr", "svm"),
                         test_fraction = 0.25, gammas = NULL, lambdas = 10^(-4:2),
                         costs = 10^(-1:3), balanced = NULL, seed = 1, output_dir = NULL,
                         plot = interactive()) {
  method <- match.arg(method)
  data <- as.data.frame(data)
  if (!target %in% names(data)) stop(sprintf("target column '%s' not found", target), call. = FALSE)
  if (is.null(features)) features <- setdiff(names(data), target)
  bad <- setdiff(features, names(data))
  if (length(bad)) stop("feature columns not found: ", paste(bad, collapse = ", "), call. = FALSE)
  notes <- character()
  keep <- stats::complete.cases(data[c(target, features)])
  if (any(!keep)) notes <- c(notes, sprintf("%d row(s) with missing values were dropped.", sum(!keep)))
  data <- data[keep, , drop = FALSE]
  y <- data[[target]]
  classification <- is.factor(y) || is.character(y) || is.logical(y)
  if (classification) y <- droplevels(as.factor(y))
  if (!classification && length(unique(y)) <= 5) {
    notes <- c(notes, sprintf(
      "The numeric target has only %d distinct values; convert it with factor() for classification.",
      length(unique(y))))
  }
  enc <- .encode(data[features])
  X <- enc$X
  is_test <- .split(y, test_fraction, classification, seed)
  if (sum(is_test) < 10) notes <- c(notes, "Fewer than 10 test rows: test metrics are very uncertain.")
  Xtr <- X[!is_test, , drop = FALSE]
  ytr <- y[!is_test]
  Xte <- X[is_test, , drop = FALSE]
  yte <- y[is_test]
  if (classification && is.null(balanced)) {
    counts <- table(ytr)
    balanced <- max(counts) / min(counts) > 1.5
  }
  if (isTRUE(balanced)) notes <- c(notes, "Classes are imbalanced: tuned for balanced accuracy.")

  tuned <- if (method == "krr") {
    krr_tune(Xtr, ytr, gammas = gammas, lambdas = lambdas, balanced = isTRUE(balanced))
  } else {
    svm_tune(Xtr, ytr, gammas = gammas, costs = costs, seed = seed)
  }
  # For the SVM, predict through the svm_tune object so features are scaled
  # exactly as in training; kernel ridge models scale internally.
  model <- if (method == "krr") tuned$model else tuned
  pred <- stats::predict(model, Xte)
  if (method == "svm" && !classification) pred <- as.numeric(pred)
  metrics <- .metrics(yte, pred, classification)
  baseline <- if (classification) {
    maj <- names(which.max(table(ytr)))
    data.frame(baseline = sprintf("always predict '%s'", maj), accuracy = mean(yte == maj))
  } else {
    data.frame(baseline = "predict the training mean", RMSE = sqrt(mean((yte - mean(ytr))^2)))
  }

  hp <- tuned$grid
  pen <- if (method == "krr") "lambda" else "cost"
  if (tuned$best$gamma %in% range(hp$gamma)) notes <- c(notes, sprintf(
    "Best gamma (%.3g) is at the edge of the grid; widen 'gammas'.", tuned$best$gamma))
  if (tuned$best[[pen]] %in% range(hp[[pen]])) notes <- c(notes, sprintf(
    "Best %s (%.3g) is at the edge of the grid; consider widening it.", pen, tuned$best[[pen]]))
  train_pred <- stats::predict(model, Xtr)
  if (classification) {
    tr_acc <- mean(train_pred == ytr)
    te_acc <- mean(pred == yte)
    if (tr_acc - te_acc > 0.1) notes <- c(notes, sprintf(
      "Training accuracy (%.3f) is much higher than test accuracy (%.3f): possible overfitting.",
      tr_acc, te_acc))
  }

  out <- structure(list(
    task = if (classification) "classification" else "regression", method = method,
    target = target, features = features, encoded_features = colnames(X), encoding = enc$spec,
    tuning = tuned, best = tuned$best, model = model, metrics = metrics$summary,
    per_class = metrics$per_class, confusion = metrics$confusion, baseline = baseline,
    test = data.frame(row = which(is_test), actual = yte, predicted = pred),
    n_train = sum(!is_test), n_test = sum(is_test), notes = notes, data = data, files = character()
  ), class = "rbf_workflow")
  if (plot) plot(out)
  if (!is.null(output_dir)) out$files <- .write_rbf_workflow(out, output_dir)
  out
}

#' @rdname rbf_workflow
#' @param object,x An `"rbf_workflow"` object.
#' @param newdata A data frame with the feature columns.
#' @param ... Unused.
#' @export
predict.rbf_workflow <- function(object, newdata, ...) {
  X <- .encode(as.data.frame(newdata), object$encoding)$X
  p <- stats::predict(object$model, X)
  if (object$method == "svm" && object$task == "regression") as.numeric(p) else p
}

#' @rdname rbf_workflow
#' @export
print.rbf_workflow <- function(x, ...) {
  cat(sprintf("RBF kernel workflow: %s of '%s' with %s\n", x$task, x$target,
              if (x$method == "krr") "kernel ridge" else "an SVM"))
  cat(sprintf("  %d features (%d after encoding); %d training rows, %d test rows\n",
              length(x$features), length(x$encoded_features), x$n_train, x$n_test))
  pen <- if (x$method == "krr") "lambda" else "cost"
  cat(sprintf("  tuned by %s: gamma = %.4g (sigma = %.3g), %s = %.4g\n",
              if (x$method == "krr") "exact leave-one-out" else "k-fold CV",
              x$best$gamma, gamma_to_sigma(x$best$gamma), pen, x$best[[pen]]))
  cat("\nTest-set performance:\n")
  m <- x$metrics
  m$value <- signif(m$value, 4)
  print(m, row.names = FALSE)
  cat("\nBaseline:\n")
  print(x$baseline, row.names = FALSE, digits = 4)
  if (!is.null(x$confusion)) {
    cat("\nConfusion matrix (test set):\n")
    print(x$confusion)
  }
  if (length(x$notes)) cat("\nNotes:\n", paste0("  - ", x$notes, "\n"), sep = "")
  if (length(x$files)) cat("\nFiles written:\n", paste0("  ", x$files, "\n"), sep = "")
  invisible(x)
}

#' @rdname rbf_workflow
#' @export
plot.rbf_workflow <- function(x, ...) {
  two_d <- length(x$encoded_features) == 2L && x$task == "classification"
  op <- graphics::par(mfrow = c(1, if (two_d) 3 else 2), mar = c(4.2, 4.2, 3, 1))
  on.exit(graphics::par(op))
  if (x$method == "krr") {
    plot(x$tuning)
  } else {
    g <- x$tuning$grid
    graphics::plot(log10(g$gamma), g[[3]], col = as.factor(g$cost), pch = 19,
                   xlab = "log10(gamma)", ylab = x$tuning$metric, main = "Cross-validation")
  }
  if (x$task == "classification") {
    cm <- unclass(x$confusion)
    prop <- sweep(cm, 2, pmax(colSums(cm), 1), "/")
    k <- ncol(cm)
    graphics::image(seq_len(k), seq_len(k), t(prop)[, k:1, drop = FALSE], axes = FALSE,
                    col = grDevices::hcl.colors(30, "Blues", rev = TRUE), zlim = c(0, 1),
                    xlab = "actual", ylab = "predicted", main = "Test confusion matrix")
    graphics::axis(1, seq_len(k), colnames(cm), cex.axis = 0.8)
    graphics::axis(2, seq_len(k), rev(rownames(cm)), cex.axis = 0.8)
    for (i in seq_len(k)) for (j in seq_len(k)) {
      graphics::text(j, k - i + 1, cm[i, j], col = if (prop[i, j] > 0.5) "white" else "black")
    }
  } else {
    graphics::plot(x$test$actual, x$test$predicted, pch = 19, col = "#1c64d1",
                   xlab = "actual", ylab = "predicted", main = "Test set: predicted vs actual")
    graphics::abline(0, 1, lty = 2)
  }
  if (two_d) {
    f <- x$features
    plot_decision_boundary(x$model, x$data[[f[1]]], x$data[[f[2]]], x$data[[x$target]],
                           xlab = f[1], ylab = f[2])
  }
  invisible(x)
}

.write_rbf_workflow <- function(wf, dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  files <- c(metrics = file.path(dir, "test_metrics.csv"),
             tuning = file.path(dir, "tuning_grid.csv"),
             predictions = file.path(dir, "test_predictions.csv"),
             figure = file.path(dir, "workflow_plots.png"),
             report = file.path(dir, "report.txt"))
  utils::write.csv(wf$metrics, files[["metrics"]], row.names = FALSE)
  utils::write.csv(wf$tuning$grid, files[["tuning"]], row.names = FALSE)
  utils::write.csv(wf$test, files[["predictions"]], row.names = FALSE)
  if (!is.null(wf$per_class)) {
    files <- c(files, per_class = file.path(dir, "per_class_metrics.csv"),
               confusion = file.path(dir, "confusion_matrix.csv"))
    utils::write.csv(wf$per_class, files[["per_class"]], row.names = FALSE)
    utils::write.csv(as.data.frame.matrix(wf$confusion), files[["confusion"]])
  }
  grDevices::png(files[["figure"]], width = 1800, height = 700, res = 140)
  tryCatch(plot(wf), finally = grDevices::dev.off())
  writeLines(utils::capture.output(print(wf)), files[["report"]])
  files
}

#' Plot a classifier's decision regions in two dimensions
#'
#' @param model A fitted [krr()] classifier (or any model with a `predict`
#'   method returning classes) trained on two features.
#' @param x1,x2 The two feature vectors.
#' @param class Factor of true classes (for the points).
#' @param resolution Grid size per axis.
#' @param xlab,ylab Axis labels.
#' @param main Title.
#' @return Invisibly, the grid of predicted classes.
#' @examples
#' moons <- make_moons(200, noise = 0.25)
#' fit <- krr(moons[, 1:2], moons$class, gamma = 1, lambda = 0.1)
#' plot_decision_boundary(fit, moons$x1, moons$x2, moons$class)
#' @export
plot_decision_boundary <- function(model, x1, x2, class, resolution = 150, xlab = "x1",
                                   ylab = "x2", main = "Decision regions") {
  class <- as.factor(class)
  pad1 <- 0.05 * diff(range(x1))
  pad2 <- 0.05 * diff(range(x2))
  g1 <- seq(min(x1) - pad1, max(x1) + pad1, length.out = resolution)
  g2 <- seq(min(x2) - pad2, max(x2) + pad2, length.out = resolution)
  grid <- expand.grid(a = g1, b = g2)
  names(grid) <- if (!is.null(model$features) && length(model$features) == 2) model$features else c("x1", "x2")
  pred <- factor(stats::predict(model, grid), levels = levels(class))
  k <- nlevels(class)
  pal <- grDevices::hcl.colors(max(k, 3), "Dark 3")[seq_len(k)]
  graphics::image(g1, g2, matrix(as.integer(pred), resolution), col = grDevices::adjustcolor(pal, 0.25),
                  xlab = xlab, ylab = ylab, main = main, useRaster = TRUE)
  graphics::contour(g1, g2, matrix(as.integer(pred), resolution), add = TRUE, drawlabels = FALSE,
                    levels = seq_len(k - 1) + 0.5, lwd = 2)
  graphics::points(x1, x2, pch = 21, bg = pal[as.integer(class)], col = "white", cex = 0.9)
  invisible(pred)
}
