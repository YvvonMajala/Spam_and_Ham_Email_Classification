# Predicting spam or Ham emails
The code is used to predict whether an email is spam or not
# Requirements
- Have python installed 
# Importing the necessary libraries
- pandas and numpy for data manipulation.
- train_test_split from sklearn.model_selection to split the dataset into training and testing sets.
- CountVectorizer from sklearn.feature_extraction.text to convert text data into a vectorized matrix.
- classification_report from sklearn.metrics to generate a classification report for evaluating the model's performance.
- MultinomialNB from sklearn.naive_bayes to use the Multinomial Naive Bayes classifier.
# The dataset
The dataset is in the form of one csv file.
# models used

### Geetting value counts of the category column
Counts the occurrences of each unique value in the 'Category' column of df
### Creating a new column 'spam':
Adds a new column 'spam' to the DataFrame df based on the 'Category' column values.
The 'spam' column is assigned the value 1 if the corresponding 'Category' value is 'spam', and 0 otherwise.
### Splitting the dataset into training and testing sets:
Splits the 'Message' column of df as the input features (X) and the 'Category' column as the target variable (y).
Uses train_test_split to split X and y into training and testing sets (X_train, X_test, y_train, y_test) with a test size of 0.25 (25% of the data is used for testing).
### Vectorizing the training and testing data:
Initializes a CountVectorizer object as v.
Converts the training data and testing data into a vectorized matrices using the CountVectorizer
### Training the model:
Initializes a Multinomial Naive Bayes model as model.
Fits the training data and corresponding labels to the model.
### Making predictions on the testing data:
Uses the trained model (model) to predict the labels for the testing data
Generates a classification report by comparing the predicted labels with the actual labels
The classification report provides evaluation metrics such as precision, recall, F1-score, and support for each class.


