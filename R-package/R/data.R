#' Wisconsin Diagnostic Breast Cancer data
#'
#' Features computed from digitised images of fine-needle aspirates of breast
#' masses, with the diagnosis. 357 benign and 212 malignant tumours.
#'
#' @format A data frame with 569 rows and 31 columns: `diagnosis` (factor:
#'   `benign`, `malignant`) and 30 numeric features: the mean, standard error
#'   and worst (largest) value of radius, texture, perimeter, area,
#'   smoothness, compactness, concavity, concave points, symmetry and fractal
#'   dimension.
#' @source Wolberg, W., Mangasarian, O., Street, N. and Street, W. (1993).
#'   Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository,
#'   \doi{10.24432/C5DW2B}. Licensed CC BY 4.0.
#' @examples
#' table(breast_cancer$diagnosis)
#' rbf_workflow(breast_cancer, "diagnosis", plot = FALSE)$metrics
"breast_cancer"
