from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
from tensorflow.keras.regularizers import L2

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, classification_report
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, auc

from keras.models import load_model



app = Flask(__name__)

new_model = load_model(r"C:\Users\timon\OneDrive\Desktop\ML-Project\Wine_classifier_Flask_app\predict_credit.h5")

cols= ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():

    #for production do some try -> except error checking...
    
    if request.method == 'POST':
        int_features = [x for x in request.form.values()]
        final = np.array(int_features, dtype=np.float32)
        data_unseen = pd.DataFrame([final], columns=cols)
        data_unseen = np.array(data_unseen)
        prediction = new_model.predict(data_unseen.reshape(1,11))
        prediction = np.argmax(prediction,axis=1)

        def predict_prob(number):
            return [number[0],1-number[0]]
    

    if prediction == 0:
        y_prob = np.array(list(map(predict_prob, new_model.predict(data_unseen.reshape(1,11)))))
        proba = y_prob.item(0)*100
        prediction = "bad"
    else:
        y_prob = np.array(list(map(predict_prob, new_model.predict(data_unseen.reshape(1,11)))))
        proba = y_prob.item(1)*100
        prediction = "good."
    
    return render_template('home.html', pred='The expected wine quality will be {}, with '.format(prediction) + str(round(proba,2)) + '% accuracy.')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = new_model.predict(data_unseen.reshape(1,11))
    output = np.argmax(prediction,axis=1)
    return jsonify(output)

if __name__=='__main__':
    app.run(host = '0.0.0.0', debug=True)










# def predict(model, list_value):

#     list_value = np.array(dataframe)

#     predict_x = model.predict(list_value.reshape(1,11))

#     classes_x=np.argmax(predict_x,axis=1)

#     return classes_x

# if __name__=="__main__":
#     list_value = [[7.3,0.65,0.00,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0]]
#     column_value = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol']
#     dataframe = pd.DataFrame(list_value, columns = column_value)
    	

#     prdct = predict(new_model, dataframe)
#     print(prdct)