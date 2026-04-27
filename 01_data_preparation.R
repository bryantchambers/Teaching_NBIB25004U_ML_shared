#' 01_Data_Preparation.R
#' 
#' This script performs initial data cleaning and preprocessing for microbial metagenomics 
#' machine learning analysis. It covers the first phase of the lab practical:
#' Data Preparation and Feature Engineering.
#'
#' Steps:
#' 1. Cleaning Names and Columns
#' 2. Standardizing Data Types
#' 3. Handling Missing Values, Variance, and Correlation (via mikropml)
#' 4. Normalization and Scaling (via mikropml)

# Load configuration and dependencies
source("/src/config.R")
library(mikropml)
library(dplyr)
library(janitor)

# 0. Setup Logging and Seed
log_file <- file.path(logs_dir, "01_data_preparation.log")
sink(log_file, append = FALSE, split = TRUE)
set.seed(seed)

cat("Starting Data Preparation Phase...\n")
cat("Using random seed:", seed, "\n")

# 1. Cleaning Names and Columns
# Novice Tip: Special characters or spaces in column names can break R functions.
# We use janitor::clean_names() to ensure all headers are snake_case and ASCII-safe.
cat("Step 1: Cleaning column names...\n")
cleaned_data <- raw_data %>%
  clean_names()

# 2. Standardizing Data Types
# Novice Tip: Models require numeric inputs. We ensure all OTU columns are numeric.
# In this example dataset they already are, but we apply this as a best practice.
cat("Step 2: Ensuring numeric data types for features...\n")
cleaned_data <- cleaned_data %>%
  mutate(across(-dx, as.numeric))

# 3-8. Advanced Preprocessing with mikropml
# mikropml::preprocess_data() is a powerful function that automates:
# - Removing observations with missing outcomes
# - Imputing missing values (Median/Mean)
# - Removing Zero/Near-Zero Variance features
# - Collapsing highly correlated features (r > 0.90)
# - Normalization and Scaling (Z-score)
# - One-Hot Encoding for categorical variables
cat("Step 3-8: Running mikropml::preprocess_data()...\n")
# We specify 'dx' as our outcome column.
preprocess_results <- preprocess_data(
  dataset = cleaned_data,
  outcome_colname = "dx",
  method = c("center", "scale"), # Z-score scaling
  remove_var = "nzv",            # Remove near-zero variance
  collapse_corr_feats = TRUE      # Remove redundant features
)

# Extract the transformed data
final_data <- preprocess_results$dat_transformed

# Summary of changes
cat("\n--- Preprocessing Summary ---\n")
cat("Original feature count:", ncol(cleaned_data) - 1, "\n")
cat("Final feature count:", ncol(final_data) - 1, "\n")
cat("Features removed due to low variance or correlation:", 
    length(preprocess_results$removed_feats), "\n")

# Save intermediate output
output_file <- file.path(results_dir, "cleaned_data.rds")
saveRDS(preprocess_results, file = output_file)

cat("\nSuccess: Cleaned data and results object saved to:", output_file, "\n")
cat("Data Preparation Phase Complete.\n")

sink()
