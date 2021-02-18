import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

""" 
Uploading the models generated

"""
model = pickle.load(open("model.pkl","rb"))
transformed = pickle.load(open("transform.pkl","rb"))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method =="POST":
        message = request.form["Enter your message"]
        data = [message]
        text_vector = transformed.transform(data).toarray()
        prediction = model.predict(text_vector)

        if prediction == [0]:
            Result = " It is a Ham Message"
        else:
            Result = " Stay AWAY it is a SPAM"


    return render_template('index.html', prediction_text=Result)


if __name__ == "__main__":
    app.run(debug=True)