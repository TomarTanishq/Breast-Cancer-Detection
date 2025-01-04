#   Importing Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#   Initializing variables to store file location
wdbc_names_fileloc='D:\PROJECTS\Breast Cancer Detection\Dataset\wdbc.names'
wdbc_data_fileloc='D:\PROJECTS\Breast Cancer Detection\Dataset\wdbc.data'

#   Store feature names 
column_names=["ID", "Diagnosis"] + [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness", 
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", 
    "Mean Fractal Dimension", "Radius SE", "Texture SE", "Perimeter SE", "Area SE", 
    "Smoothness SE", "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", 
    "Fractal Dimension SE", "Worst Radius", "Worst Texture", "Worst Perimeter", 
    "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity", 
    "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

#   Load data withe column names
data=pd.read_csv(wdbc_data_fileloc,header=None,names=column_names)

#   Drop ID column
data.drop(columns=['ID'],inplace=True)

#   Encode target variables(M->1,B->0)
data['Diagnosis']=data["Diagnosis"].map({'M':1,'B':0})

#   Splitting data(Seperating target and features from each other)
X=data.drop(columns=['Diagnosis'])  #   Features
Y=data['Diagnosis'] #   Target

#   Splitting dataset into Training and Testing Sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#   Standardize the feature values
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#   Train a LogisticRegression model
model=LogisticRegression()
model.fit(X_train_scaled,Y_train)

#   Make predictions and evaluate training model
X_train_prediction=model.predict(X_train_scaled)
X_train_accuracy=accuracy_score(Y_train,X_train_prediction)
print("Training model accuracy ",X_train_accuracy)

#   Make predicitons and evaluate Testing model
X_test_prediction=model.predict(X_test_scaled)
X_test_accuracy=accuracy_score(Y_test,X_test_prediction)
print("Testing model accuracy: ",X_test_accuracy)

#   Input Data manually to check whether tumor is benign or malignant
input_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885
            ,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315
            ,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

#   Change input data to numpy array for calculations
input_data_as_numpy_array=np.asarray(input_data)

#   Reshape numpy array as we are predicting for one data point
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1) #   Reshape this array to 2D array with 1 row and as many columns as needed(30 columns in this case)

#   Standardize or Scale the input data 
input_data_scaled=scaler.transform(input_data_reshaped)

#   Prediction based on new input data
prediction=model.predict(input_data_scaled)
if(prediction[0]==1):
    print("The tumor might be cancerous,please refer to a specialist as soon as possible")
else:
    print("The tumor might not be cancerous,still refer to a specialist")