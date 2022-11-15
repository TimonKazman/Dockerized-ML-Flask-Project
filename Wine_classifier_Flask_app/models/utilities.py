import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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

scaler = MinMaxScaler()

new_model = load_model('models/predict_credit.h5')

def predict(model, list_value):

    list_value = np.array(dataframe)

    predict_x = model.predict(list_value.reshape(1,11))

    classes_x=np.argmax(predict_x,axis=1)

    return classes_x

if __name__=="__main__":
    list_value = [[7.3,0.65,0.00,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0]]
    column_value = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
    dataframe = pd.DataFrame(list_value, columns = column_value)
    	

    prdct = predict(new_model, dataframe)
    print(prdct)