

###Importing of libraries

#data manipulation libraries
import pandas as pd
import numpy as np

#importing train and test splitting libraries
from sklearn.model_selection import train_test_split

#importing a text vectorizer form sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

"""**Reading the dataset**"""

#reading the dataset
df = pd.read_csv('/content/drive/MyDrive/NLP CHATBOT/SPAM AND HAM EMAILS CLASSIFICATION/spam.csv')
#printing the first 5 rows of the dataset
df.head()

#Getting the value counts of the category column
df['Category'].value_counts()

"""- The dataset is imbalanced"""

#changing the categorical column to 0 and 1 values
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
#checking the head of the dataset
df.head()

"""- A new column from the Category column known as spam has been created"""

#splitting the dataset into trainjing and testing set
X_train,X_test, y_train,y_test = train_test_split( df.Message, df.Category, test_size = 0.25)
X_train.head()

#getting the shape of the training and testing set
print('shape of X_train is: ', X_train.shape)
print('shape of X_test is: ', X_test.shape)

type(X_train.values)

print(X_train[:4])
print(type(X_train))

#converting X_train to a vectorised matrix
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()

X_train_cv = v.fit_transform(X_train.values)
X_train_cv

#converting x_test to a vectorised matrix
X_test_cv = v.transform(X_test.values)
X_test_cv

#convering it to an array
X_train_cv.toarray()[:2][0]

#converting X_train to an array
X_train_np = X_train_cv.toarray()
X_train_np[0]

#converting X_test to an array
X_test_np = X_test_cv.toarray()
X_test_np[0]

#using a naives bayes classifier
from sklearn.naive_bayes import MultinomialNB
#INITIALIZING THE MODEL
model = MultinomialNB()
#fitting the data
model.fit(X_train_cv, y_train)

#Making predictions on the x_test
preds = model.predict(X_test_cv)
report = classification_report(preds, y_test)
print(report)