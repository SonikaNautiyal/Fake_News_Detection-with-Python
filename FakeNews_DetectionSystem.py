#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[19]:


#Read the data
df = pd.read_csv("C:/Users/nauti/OneDrive/Desktop/DATA/TimesInternetNewsData.csv")

#Get shape and head
df.shape
df.head()


# In[20]:


#Get the labels
labels=df.label
labels.head()


# In[21]:


#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[22]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[23]:


#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[25]:


from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set
y_pred = pac.predict(tfidf_test)

# Calculate accuracy
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

# Convert string labels to numeric values for R² score calculation
y_test_numeric = [1 if label == 'REAL' else 0 for label in y_test]
y_pred_numeric = [1 if label == 'REAL' else 0 for label in y_pred]

# Calculate R² score
r2 = r2_score(y_test_numeric, y_pred_numeric)
print(f'R² Score: {r2}')


# In[26]:


#Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example labels and predictions
# Replace these with your actual `y_test` and `y_pred`
y_test = ['FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE']
y_pred = ['FAKE', 'REAL', 'REAL', 'FAKE', 'FAKE', 'REAL', 'REAL']

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:





# In[ ]:




