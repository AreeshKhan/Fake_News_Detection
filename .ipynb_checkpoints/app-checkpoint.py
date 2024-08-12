import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Import vectoriser model and Gradient Boosting Classifier model
vectoriser_model = pickle.load(open('vectorization.pkl', 'rb'))
GradientBoostingClassifier_model = pickle.load(open('GradientBoostingClassifier.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        text = request.form.get('text')
        new_data_vectorised = vectoriser_model.transform([text])
        result = GradientBoostingClassifier_model.predict(new_data_vectorised)
        if result == 0:
            result = 'True'
        else:
            result = 'Fake'
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")















# import pickle
# from flask import Flask,request,jsonify,render_template # type: ignore
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)
# #Import ridge regressor model and standard scaler pickle
# vectoriser_model=pickle.load(open('vectorization.pkl','rb'))
# GradientBoostingClassifier_model=pickle.load(open('GradientBoostingClassifier.pkl','rb'))

# #Route for home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predictdata',methods=['GET','POST'])
# def predict_datapoint():
#     if request.method=='POST':
#         text=float(request.form.get('text'))
        
#         new_data_vectorised=vectoriser_model.transform([[text]])
#         result=GradientBoostingClassifier_model.predict(new_data_vectorised)
#         if (result==0){
#             result='True'
#         }
#         else{
#             result='Fake'
#         }

#         return render_template('home.html',result=result)

#     else:
#         return render_template('home.html')

# if __name__=="__main__":
#     app.run(host="0.0.0.0")