"""Mental Health Prediction Using Sentiment Analysis .ipynb

Importing dataset into dataframe

---
"""

import nltk
import pandas as pd
dataset = pd.read_csv(r'/content/drive/MyDrive/NLP/NLP Project/mental_health.csv')

from google.colab import drive
drive.mount('/content/drive')

#testing if .csv is imported and showing structure of dataset

print(dataset.info())
dataset.head()

#2. Print column 'text'

print(dataset['text'])

# Check for any missing values in the dataset
print("\nMissing Values:")
print(dataset.isnull().sum())

import matplotlib.pyplot as plt

# Explore the distribution of labels (0 or 1)
label_distribution = dataset['label'].value_counts()
print("\nLabel Distribution:")
print(label_distribution)

# Visualize label distribution using simple plot
label_distribution.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()



#9. create a list of sentances

dataset_text = (dataset['text']).to_list()
print(dataset_text[:1000])



#10. join all sentances into one big text

raw_dataset_text = " ".join(dataset_text)
raw_dataset_text[:1000]

#now since we have raw text we want to have list of words from the previous test

words = raw_dataset_text.split(" ")
print(words[:300])

#print how many words are are in dataset and how many unique words are in dataset
print("Dataset has",len(words), "words.")
print("Dataset has", len(set(words)), "unique words.")

#print FreqDist for this dataset
import nltk
freq = nltk.FreqDist(words)
print(type(freq))

print(freq.most_common(50))

import nltk
#function to remove stopwords in text and non alpha words and return them in lowercase
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

def remove_stopwords(text):

    word_tokens = word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')

    content = [w.lower() for w in word_tokens if w.lower() not in stopwords and w.isalpha()]
    return " ".join(content)



#create new column in dataset without stopwords
dataset['without stopwords'] = [remove_stopwords(text) for text in dataset['text']]
print(dataset)

dataset_text1 = (dataset['without stopwords']).to_list()
raw_dataset_text1 = " ".join(dataset_text1)
words1 = raw_dataset_text1.split(" ")
#print how many words are are in dataset and how many unique words are in dataset
print("Dataset has",len(words1), "words after removing stopwords. ")
print("Dataset has", len(set(words1)), "unique words after removing stopwords. ")

import re
from nltk.tokenize import word_tokenize

def regexi(text):
    # Tokenize the input text
    tokenized_text = word_tokenize(text)

    # List to hold cleaned words
    cleaned_words = []

    for word in tokenized_text:

        cleaned_word = re.sub(r'[^a-zA-Z\s]', '', word) #remove special characters and punctations

        cleaned_word = cleaned_word.lower()  # Convert to lowercase

        # Enhanced regex pattern to remove specific words (i, ve, br, m)
        cleaned_word = re.sub(r'\b(?:i|ve|br|m|I|re|s)\b', '', cleaned_word)

        # Add the cleaned word to the list
        cleaned_words.append(cleaned_word)

    # Join the processed words back into a string
    return ' '.join(cleaned_words)

# Example usage
text = "I've been to the market, but I'm not sure if I can br it."
cleaned_text = regexi(text)
print(cleaned_text)

#Applying the function to a new dataset column regex
dataset['apply regexes'] = [regexi(text) for text in dataset['without stopwords']]
print(dataset)

dataset_text2 = (dataset['apply regexes']).to_list()
raw_dataset_text2 = " ".join(dataset_text2)
words2 = raw_dataset_text2.split(" ")
#print how many words are are in dataset and how many unique words are in dataset
print("Dataset has",len(words2), "words after applying regexes.")
print("Dataset has", len(set(words2)), "unique words after applying regexes.")

import matplotlib.pyplot as plt
import numpy as np


total_words_before_cleaning = len(words)
unique_words_before_cleaning = len(set(words))

total_words_after_stopwords = len(words1)
unique_words_after_stopwords = len(set(words1))

total_words_after_regex = len(words2)
unique_words_after_regex =len(set(words2))
# Data preparation for plotting
categories = ['Before Cleaning', 'After Removing Stopwords', 'After Applying Regexes']
total_words_data = [total_words_before_cleaning, total_words_after_stopwords, total_words_after_regex]
unique_words_data = [unique_words_before_cleaning, unique_words_after_stopwords, unique_words_after_regex]

# Creating bar width
bar_width = 0.35
index = np.arange(len(categories))

# Creating the plot
plt.figure(figsize=(10, 6))
plt.bar(index, total_words_data, bar_width, label='Total Words', color='b')
plt.bar(index + bar_width, unique_words_data, bar_width, label='Unique Words', color='g')

# Adding labels and title
plt.xlabel('Data Processing Stage')
plt.ylabel('Number of Words')
plt.title('Word Counts Before and After Data Cleaning')
plt.xticks(index + bar_width / 2, categories)
plt.legend()

# Displaying the plot
plt.tight_layout()
plt.show()

nltk.download('wordnet')
lemma = nltk.WordNetLemmatizer()
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

print(porter.stem('music recommendations im looking expand playlist usual genres alt pop minnesota hip hop steampunk various indie genres artists people like cavetown aliceband bug hunter penelope scott various rhymesayers willing explore new genresartists such anything generic rap the type exclusively sex drugs cool rapper is rap types pretty good pop popular couple years ago dunno technical genre name anyways anyone got music recommendations favorite artistssongs'))
print(lancaster.stem('music recommendations im looking expand playlist usual genres alt pop minnesota hip hop steampunk various indie genres artists people like cavetown aliceband bug hunter penelope scott various rhymesayers willing explore new genresartists such anything generic rap the type exclusively sex drugs cool rapper is rap types pretty good pop popular couple years ago dunno technical genre name anyways anyone got music recommendations favorite artistssongs'))
print(lemma.lemmatize("music recommendations im looking expand playlist usual genres alt pop minnesota hip hop steampunk various indie genres artists people like cavetown aliceband bug hunter penelope scott various rhymesayers willing explore new genresartists such anything generic rap the type exclusively sex drugs cool rapper is rap types pretty good pop popular couple years ago dunno technical genre name anyways anyone got music recommendations favorite artistssongs"))

#the best output using spacy
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    # Process the text using SpaCy
    doc = nlp(text)

    # Extract the lemma for each token and filter out punctuation
    lemmatized_text = [token.lemma_ for token in doc if not token.is_punct]

    return " ".join(lemmatized_text)

# Example usage
text = "music recommendations im looking expand playlist usual genres alt pop minnesota hip hop steampunk various indie genres artists people like cavetown aliceband bug hunter penelope scott various rhymesayers willing explore new genresartists such anything generic rap the type exclusively sex drugs cool rapper is rap types pretty good pop popular couple years ago dunno technical genre name anyways anyone got music recommendations favorite artistssongs"
lemmatized_text = lemmatize_text(text)
print(lemmatized_text)

#12. Apply Lemmatization on all texts - rows in dataset

dataset['v1: final text'] = [lemmatize_text(text) for text in dataset['apply regexes']]
dataset

proba = (dataset['v1: final text']).to_list()
raw_dataset_proba = "".join(proba)
raw_dataset_proba[:10000]

dataset['cleaned_text'] = [regexi(text) for text in dataset['v1: final text']]
print(dataset)

# Get data about most used words in class 1

all_texts_1 = []

for text in dataset[dataset['label']==1]['cleaned_text']:
  for word in nltk.word_tokenize(text):
    all_texts_1.append(word)

nltk.FreqDist(all_texts_1).plot(20)

# Get data about most used words in class 0

all_texts_0 = []

for text in dataset[dataset['label']==0]['cleaned_text']:
  for word in nltk.word_tokenize(text):
    all_texts_0.append(word)

nltk.FreqDist(all_texts_0).plot(20)

print(nltk.FreqDist(all_texts_1).most_common())

print(nltk.FreqDist(all_texts_0).most_common())

#Take the features - 'cleaned_text' column as X variable and class - 'label' column as Y variable

X = dataset['cleaned_text']
y = dataset['label']

print(X)
print(y)

# Now we need to transform text features into numberic values such that we will count occurance of each word in whole dataset for each row

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

tf = CountVectorizer()
x_tf = tf.fit_transform(X)

#  created dataframe just for visualization purposes, to show how our dataset looks now it has no value to the prediction

new_dataset = pd.DataFrame(x_tf.toarray(), columns=tf.get_feature_names_out())
new_dataset

# Import library from sklearn to divide dataset into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_tf, y, test_size = 0.3, stratify = y)

# Import RandomForestClassifier from sklearn library and train model with train sets

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train,y_train)

#Evaluate the accuracy of the model with test datasets

model.score(X_test, y_test)

#With Confusion Matrix visualize the predicted values, which are predicted correct which not

from sklearn.metrics import multilabel_confusion_matrix

y_pred = model.predict(X_test)
cf_matrix = multilabel_confusion_matrix(y_test, y_pred, labels=[0,1])
print(cf_matrix)

import seaborn as sns
import numpy as np

#Confusion Matrix for first class

labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix[0], annot=labels, fmt='', cmap='Blues')

#Confusion Matrix for second class

labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix[1], annot=labels, fmt='', cmap='Greens')

from sklearn.naive_bayes import MultinomialNB

# Instantiate the Naive Bayes classifier
nb_model = MultinomialNB()

# Train the model with the training data
nb_model.fit(X_train, y_train)

# Evaluate the accuracy of the model with the test data
nb_accuracy = nb_model.score(X_test, y_test)
print(f"Naive Bayes Model Accuracy: {nb_accuracy}")

# Predict the test dataset
y_pred_nb = nb_model.predict(X_test)

# Generate the confusion matrix for the Naive Bayes predictions
cf_matrix_nb = multilabel_confusion_matrix(y_test, y_pred_nb, labels=[0, 1])
print(cf_matrix_nb)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of Confusion Matrix for the first class
sns.heatmap(cf_matrix_nb[0], annot=labels, fmt='', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix for First Class')
plt.show()

# Visualization of Confusion Matrix for the second class
sns.heatmap(cf_matrix_nb[1], annot=labels, fmt='', cmap='Greens')
plt.title('Naive Bayes Confusion Matrix for Second Class')
plt.show()

# Function to preprocess user input text
def preprocess_text(text):

    cleaned_text = regexi(text)
    text_features = tf.transform([cleaned_text])  # Vectorize text
    return text_features

# Function to predict mental health status
def predict_mental_health(text):
    text_features = preprocess_text(text)
    prediction = model.predict(text_features)
    return prediction[0]

# User input
user_input = input("Enter text for mental health prediction: ")

# Predict and print the result
prediction_result = predict_mental_health(user_input)
if prediction_result == 1:
    print("The text indicates potential mental health concerns.")
else:
    print("No mental health concerns detected in the text.")

# Function to preprocess user input text
def preprocess_text(text):

    cleaned_text = regexi(text)
    text_features = tf.transform([cleaned_text])  # Vectorize text
    return text_features

# Function to predict mental health status
def predict_mental_health(text):
    text_features = preprocess_text(text)
    prediction = model.predict(text_features)
    return prediction[0]

# User input
user_input = input("Enter text for mental health prediction: ")

# Predict and print the result
prediction_result = predict_mental_health(user_input)
if prediction_result == 1:
    print("The text indicates potential mental health concerns.")
else:
    print("No mental health concerns detected in the text.")