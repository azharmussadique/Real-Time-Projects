# Load necessary libraries
library(ggplot2)
library(dplyr)
library(readxl)
library(tidyverse)
library(corrplot)
library(leaflet)
library(caTools)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(stringr)
library(text2vec)
library(Matrix)
library(e1071)  # For Naive Bayes
library(caret)   # For train-test split and evaluation
library(randomForest)

# Read and Prepare Dataset
data <- read.csv("WELFake_Dataset.csv")
df <- data.frame(data)

# Dataset Summary and Label Distribution
summary(df)
per_real <- (sum(df[["label"]] == "1") / nrow(df)) * 100
per_fake <- (sum(df[["label"]] == "0") / nrow(df)) * 100
cat("The percentage of real news is: ", per_real)
cat("The percentage of fake news is: ", per_fake)

# Data Cleaning and NA Handling
df <- df[, !grepl("^Unnamed|^X", names(df))]
count_na <- sapply(df, function(col) sum(is.na(col) | col == ""))
cat("Number of rows with NA values:", count_na, "\n")
na_df <- data.frame(Column = names(count_na), NA_Count = count_na)

# Plot NA Counts per Column
ggplot(na_df, aes(x = Column, y = NA_Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Number of NA Values per Column", x = "Columns", y = "Count of NA Values") +
  theme_minimal()

# Remove Empty Strings from Dataset
remove_empty_strings <- function(df) {
  df_clean <- df[!apply(df, 1, function(row) any(row == "")), ]
  return(df_clean)
}
df <- remove_empty_strings(df)

# Add Title and Text Length Feature
df[["title+text"]] <- paste(df[["title"]], df[["text"]])
df[["length"]] <- nchar(df[["title+text"]]) - str_count(df[["title+text"]], " ")

# Body Length Distribution Plot
ggplot(df, aes(x = length, fill = factor(label))) +
  geom_histogram(binwidth = (200 - 0) / 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("0" = "#FF5733", "1" = "#33FFB8"), 
                    labels = c("0" = "Fake", "1" = "Real")) +
  labs(title = "Body Length Distribution by Label",
       x = "Body Length", y = "Frequency", fill = "Label") +
  theme_minimal() +
  theme(legend.position = "top") +  
  scale_x_continuous(limits = c(0, 25000), breaks = seq(0, 5000, by = 2000)) +  
  scale_y_continuous(limits = c(0, 50), breaks = seq(0, 50, by = 10))

# Number of Fake and Real News Bar Plot
ggplot(df, aes(x= label, fill = factor(label))) +
  geom_bar() +
  labs(title = "Fake Vs Real News")

# Data Splitting for Train and Test Sets
set.seed(53)
X <- df$text
y <- df$label
split <- sample.split(y, SplitRatio = 0.80)
X_train <- subset(X, split == TRUE)
X_test <- subset(X, split == FALSE)
y_train <- subset(y, split == TRUE)
y_test <- subset(y, split == FALSE)

# Wordcloud for Titles
titles <- paste(df$title, collapse = " ")
wordcloud(titles, max.words = 300, random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"), scale = c(3, 0.5), min.freq = 2)

# Wordcloud for Fake and Real News Texts
fake_news <- X_train[y_train == 0]
real_news <- X_train[y_train == 1]
fake_texts <- paste(fake_news, collapse = " ")
real_texts <- paste(real_news, collapse = " ")

wordcloud(fake_texts, max.words = 300, random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"), scale = c(3, 0.5), min.freq = 2)
wordcloud(real_texts, max.words = 300, random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"), scale = c(3, 0.5), min.freq = 2)

# Bag of Words Model: Preprocessing and Term-Document Matrices
corpus_train <- Corpus(VectorSource(X_train))
corpus_train <- tm_map(corpus_train, content_transformer(tolower))
corpus_train <- tm_map(corpus_train, removePunctuation)
corpus_train <- tm_map(corpus_train, removeNumbers)
corpus_train <- tm_map(corpus_train, removeWords, stopwords("SMART"))
corpus_train <- tm_map(corpus_train, stripWhitespace)
dtm_train <- DocumentTermMatrix(corpus_train)

# Remove Sparse Terms and Prepare Training Data
dtm_train_sparse <- removeSparseTerms(dtm_train, sparse = 0.90)
train_matrix <- as.matrix(dtm_train_sparse)
train_df <- as.data.frame(train_matrix)
train_df$label <- factor(y_train)

# Preprocess Test Data and Align with Training DTM
corpus_test <- Corpus(VectorSource(X_test))
corpus_test <- tm_map(corpus_test, content_transformer(tolower))
corpus_test <- tm_map(corpus_test, removePunctuation)

corpus_test <- tm_map(corpus_test, removeNumbers)
corpus_test <- tm_map(corpus_test, removeWords, stopwords("SMART"))
corpus_test <- tm_map(corpus_test, stripWhitespace)
dtm_test <- DocumentTermMatrix(corpus_test)
dtm_test_aligned <- dtm_test[, Terms(dtm_train_sparse)]
test_matrix <- as.matrix(dtm_test_aligned)
test_df <- as.data.frame(test_matrix)
test_df$label <- factor(y_test)

# Naive Bayes Model Training and Evaluation
model <- naiveBayes(label ~ ., data = train_df)
predictions <- predict(model, test_df)
confusion_matrix <- confusionMatrix(predictions, test_df$label)
print(confusion_matrix)
accuracy <- confusion_matrix$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy, 2) * 100))

# Random Forest Model Training and Evaluation
# Set the seed for reproducibility
set.seed(123)

# Ensure unique and valid column names in training and test datasets
colnames(train_df) <- make.names(colnames(train_df), unique = TRUE)
colnames(train_df) <- gsub("[^a-zA-Z0-9_]", "", colnames(train_df))
colnames(test_df) <- make.names(colnames(test_df), unique = TRUE)

# Ensure the label column is a factor in the training dataset
train_df$label <- as.factor(train_df$label)

# Check for duplicated columns and remove them from both train and test data
# Remove any problematic or unwanted columns like "X1"
train_df <- train_df[, !(colnames(train_df) %in% c("X1"))]
test_df <- test_df[, !(colnames(test_df) %in% c("X1"))]

# Ensure that both training and testing datasets have the same column names after cleaning
common_columns <- intersect(colnames(train_df), colnames(test_df))
train_df <- train_df[, common_columns]
test_df <- test_df[, common_columns]

# Train the Random Forest model
rf_model <- randomForest(label ~ ., data = train_df, ntree = 300)
print(rf_model)

# Predict on the test data
predictions_rf <- predict(rf_model, test_df)

# Confusion Matrix
library(caret)
conf_matrix_rf <- confusionMatrix(predictions_rf, test_df$label)
print(conf_matrix_rf)

# Calculate and print the accuracy of the model
accuracy_rf <- conf_matrix_rf$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy_rf, 2) * 100, "%"))
