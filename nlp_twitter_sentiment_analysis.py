# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Load the training dataset
dataset_train = pd.read_csv('/content/twitter_training.csv', header=None)  # No header in the original file
dataset_train.shape

# Load the validation dataset
dataset_val =pd.read_csv('/content/twitter_validation.csv', header=None)  # No header in the original file
dataset_val.shape

dataset_train.head()

dataset_train.tail()

dataset_val.head()

dataset_val.tail()

# Define the header
header = ["Twitter ID", "Source", "Label","Comment"]

# Adding the header
dataset_train.columns = header
dataset_val.columns=header

# Save the file with the new header
dataset_train.to_csv('twitter_training_data_with_header.csv', index=False)
dataset_val.to_csv('twitter_validation_data_with_header.csv', index=False)

dataset_train.head(5)

dataset_train.info()

dataset_val.head(5)

dataset_val.info(5)

"""#Exploratory Data Analysis"""

# check statistical measures
dataset_train.describe(include = 'all')

# check for null values
dataset_train.isnull().sum()

# check unique values
dataset_train.nunique()

dataset_train.nunique()['Label']

dataset_train.value_counts('Label')

dataset_train.value_counts('Label').plot(kind='bar')

"""#Data visualization"""

# Calculate the number of sentiment labels in training data and validation data
train_sentiment_counts = dataset_train['Label'].value_counts()
valid_sentiment_counts = dataset_val['Label'].value_counts()

# Draw a pie chart for the training data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(train_sentiment_counts, labels=train_sentiment_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Training Data Sentiment Distribution')

# Draw a pie chart for the valid data
plt.subplot(1, 2, 2)
plt.pie(valid_sentiment_counts, labels=valid_sentiment_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Validation Data Sentiment Distribution')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.bar(train_sentiment_counts.index, train_sentiment_counts.values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Training Data Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Display quantity in each column
for i, count in enumerate(train_sentiment_counts.values):
    plt.text(x=i, y=count, s=str(count), ha='center', va='bottom')


plt.subplot(1, 2, 2)
plt.bar(valid_sentiment_counts.index, valid_sentiment_counts.values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Validation Data Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Display quantity in each column
for i, count in enumerate(valid_sentiment_counts.values):
    plt.text(x=i, y=count, s=str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()


dataset_train.value_counts('Comment')

train_df = dataset_train.dropna(subset=['Comment'])

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

print(stopwords.words('english'))

train_df.Comment[0]

word_tokenize(train_df.Comment[0])



import string

train_df['Comment'] = train_df['Comment'].astype(str)

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

train_df['Comment'] = train_df['Comment'].astype(str)
train_df['Comment'] = train_df['Comment'].apply(preprocess_text)
train_df.head(5)

dataset_val['Comment'] = dataset_val['Comment'].astype(str)
dataset_val['Comment'] = dataset_val['Comment'].apply(preprocess_text)
X_val = dataset_val['Comment']
y_val = dataset_val['Label']

X_val.head(5)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_df['Comment'], train_df['Label'], test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Ensure that X_train and X_test are of type str
X_train = X_train.astype(str)
X_test = X_test.astype(str)


# SVC Model training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Define the model pipeline
model_pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=None)), # Vectorize the data using TF-IDF
    ('svm', SVC(kernel='linear'))
])

# Train the model
model_pipeline_svc.fit(X_train, y_train)

# Predict labels for the validation data using the trained model
y_pred_val_svc = model_pipeline_svc.predict(X_val)

# Evaluate the model's performance on the validation data
accuracy_val_svc = accuracy_score(y_val, y_pred_val_svc)
report_val_svc = classification_report(y_val, y_pred_val_svc)

print("Validation Accuracy:", accuracy_val_svc)
print("Validation Classification Report:\n", report_val_svc)

# Predict on the test set
y_pred_svc = model_pipeline_svc.predict(X_test)

# Evaluate the model
accuracy_svc = accuracy_score(y_test, y_pred_svc)
report_svc = classification_report(y_test, y_pred_svc)

print("Accuracy on test data:", accuracy_svc)
print("Classification Report on test data:\n", report_svc)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# y_pred contains the predicted labels and y_test contains the true labels
accuracy_svc = accuracy_score(y_test, y_pred_svc)
precision_svc = precision_score(y_test, y_pred_svc, average='weighted')
recall_svc = recall_score(y_test, y_pred_svc, average='weighted')
f1_svc = f1_score(y_test, y_pred_svc, average='weighted')

print("Accuracy:", accuracy_svc)
print("Precision:", precision_svc)
print("Recall:", recall_svc)
print("F1-score:", f1_svc)

# Model 2: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# Define the model pipeline with Random Forest classifier
model_pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('rf', RandomForestClassifier())
])

# Train the model
model_pipeline_rf.fit(X_train, y_train)


# Predict labels for the validation data using the trained model
y_pred_val_rf = model_pipeline_rf.predict(X_val)

# Evaluate the model's performance on the validation data
accuracy_val_rf = accuracy_score(y_val, y_pred_val_rf)
report_val_rf = classification_report(y_val, y_pred_val_rf)

print("Validation Accuracy:", accuracy_val_rf)
print("Validation Classification Report:\n", report_val_rf)

# Predict on the test set
y_pred_rf = model_pipeline_rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test,y_pred_rf)

print("Accuracy:", accuracy_rf)
print("Classification Report:\n", report_rf)

# y_pred contains the predicted labels and y_test contains the true labels
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)


from joblib import dump

dump(model_pipeline_rf, 'model_NLP_sentiment_analysis.pkl')

"""Download the model file from Google Colab to local machine"""

from google.colab import files

files.download('model_NLP_sentiment_analysis.pkl')

