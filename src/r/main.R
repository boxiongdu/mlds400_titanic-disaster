# Load data
# Docker sets the working directory to /app.
data_dir <- file.path("src", "data")
train <- read.csv(file.path(data_dir, "train.csv"))

cat("Training dataset loaded successfully.\n")
cat("Rows:", nrow(train), "Columns:", ncol(train), "\n")

cat("\nFirst five rows:\n")
print(head(train, 5))

# Explore the dataset
cat("\nDataset structure:\n")
str(train)

cat("\nSummary statistics:\n")
print(summary(train))

cat("\nMissing values by column:\n")
print(colSums(is.na(train)))

cat("\nOverall survival rate:\n")
cat(sprintf("%.2f%%\n", mean(train$Survived) * 100))

cat("\nSurvival rate by sex:\n")
print(aggregate(Survived ~ Sex, data = train, FUN = mean))

cat("\nSurvival rate by passenger class:\n")
print(aggregate(Survived ~ Pclass, data = train, FUN = mean))

# Select features
features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")
prepared <- train[, c(features, "Survived")]

cat("\nSelected features:", paste(features, collapse = ", "), "\n")

# Calculate imputation values using training data only
age_median <- median(prepared$Age, na.rm = TRUE)
fare_median <- median(prepared$Fare, na.rm = TRUE)

# Use the same preprocessing for training and test data
prepare_features <- function(df) {
  df$Age[is.na(df$Age)] <- age_median
  df$Fare[is.na(df$Fare)] <- fare_median

  # Fixed levels keep categorical encoding consistent
  df$Sex <- factor(df$Sex, levels = c("male", "female"))
  df$Pclass <- factor(df$Pclass, levels = c(1, 2, 3))

  return(df)
}

prepared <- prepare_features(prepared)

cat("\nFilled missing Age values with training median:", age_median, "\n")
cat("Filled missing Fare values with training median:", fare_median, "\n")
cat("Converted Sex and Pclass to categorical factors.\n")
cat("Baseline categories: male and passenger class 1.\n")

cat("\nMissing values after preprocessing:\n")
print(colSums(is.na(prepared)))

# Fit logistic regression
model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
  data = prepared,
  family = binomial(link = "logit")
)

cat("\nLogistic regression model fitted.\n")

# Evaluate training accuracy
train_probabilities <- predict(
  model,
  newdata = prepared,
  type = "response"
)

train_predictions <- as.integer(train_probabilities >= 0.5)
train_accuracy <- mean(train_predictions == prepared$Survived)

cat(sprintf("\nTraining accuracy: %.2f%%\n", train_accuracy * 100))
cat("This accuracy is measured on the data used to fit the model.\n")

# Load and prepare test data
test <- read.csv(file.path(data_dir, "test.csv"))

cat("\nTest dataset loaded. Passengers:", nrow(test), "\n")

test_prepared <- prepare_features(test[, features])

cat("Applied training medians and matching factor levels to test data.\n")

# Predict test survival
test_probabilities <- predict(
  model,
  newdata = test_prepared,
  type = "response"
)

test_predictions <- as.integer(test_probabilities >= 0.5)

results <- data.frame(
  PassengerId = test$PassengerId,
  PredictedSurvived = test_predictions
)

cat("\nFirst 10 predictions (0 = died, 1 = survived):\n")
print(head(results, 10), row.names = FALSE)
cat("Generated predictions for", nrow(results), "passengers.\n")

# Evaluate only if true labels are available
if ("Survived" %in% names(test) && !anyNA(test$Survived)) {
  test_accuracy <- mean(test_predictions == test$Survived)
  cat(sprintf("\nTest accuracy: %.2f%%\n", test_accuracy * 100))
} else {
  cat("\nTest accuracy cannot be calculated locally.\n")
  cat("Kaggle test.csv does not include true survival labels.\n")
}

cat("\nR analysis completed.\n")