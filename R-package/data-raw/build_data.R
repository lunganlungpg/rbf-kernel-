# Builds data/breast_cancer.rda from data-raw/breast_cancer.csv
# (exported from scikit-learn's bundled copy of the UCI WDBC data).
breast_cancer <- utils::read.csv("data-raw/breast_cancer.csv")
breast_cancer$diagnosis <- factor(breast_cancer$diagnosis, levels = c("benign", "malignant"))
save(breast_cancer, file = "data/breast_cancer.rda", compress = "xz")
