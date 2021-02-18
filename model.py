import pandas as pd 
import numpy as np
import pickle 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


df = pd.read_csv("spam.csv", encoding='Latin -1')

data = df[['text','target']]

data['target'] = data['target'].map({'ham':0,'spam':1})

X = data['text']
y = data['target']

cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv,open('transform.pkl','wb'))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state = 1000)

model = MultinomialNB()
model.fit(X_train,y_train)

model.score(X_test,y_test)

pickle.dump(model, open('model.pkl','wb'))