import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    '''
    For rendering results on HTML GUI
    '''

    return render_template('index.html', prediction_text=' '.join(sentence_20))


if __name__ == "__main__":
    app.run(debug=True)