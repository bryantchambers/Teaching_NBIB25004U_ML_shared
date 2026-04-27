# Project Configuration

# Global seed for reproducibility
seed <- 42

# Input data
# In a real scenario, this might be a path to a CSV. 
# For this lab, we use the built-in mikropml dataset.
library(mikropml)
data("otu_mini_bin")
raw_data <- otu_mini_bin

# Output paths
results_dir <- "/src/results"
logs_dir <- "/src/logs"
figures_dir <- "/src/figures"
