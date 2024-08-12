import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import re
import string
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Import vectoriser model and Gradient Boosting Classifier model
vectoriser_model = pickle.load(open('models/vectorization.pkl', 'rb'))
GradientBoostingClassifier_model = pickle.load(open('models/GradientBoostingClassifier.pkl', 'rb'))
DecisionTree_model = pickle.load(open('models/DecisionTreeClassifier.pkl', 'rb'))


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        text = request.form.get('text')
        def wordopt(text):
            text = text.lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\W', ' ', text)  # Replace non-word characters with spaces
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>+', '', text)
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
            return text
        
        text = wordopt(text)

        new_data_vectorised = vectoriser_model.transform([text])
        result1 = GradientBoostingClassifier_model.predict(new_data_vectorised)
        result2 = DecisionTree_model.predict(new_data_vectorised)
        final_pred=bool(result1) and bool(result2)
        
        return render_template('home.html', result=final_pred)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")





