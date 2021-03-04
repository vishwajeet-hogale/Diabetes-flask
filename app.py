# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:41:01 2020

@author: vishw
"""

from flask import Flask,render_template,request, url_for, redirect
import pickle
import numpy as np
app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("Diabetes.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    print(prediction)

    if (prediction!=0):
        return render_template('Diabetes.html',pred='Danger.',bhai="Do something about it")
    else:
        return render_template('Diabetes.html',pred='safe.',bhai="You are safe!")





if __name__ == '__main__':
    app.run(port=5000,debug=True)
