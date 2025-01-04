import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading data from sklearn
breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
# print(breast_cancer_dataset)

#Loading data to a dataframe
data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)

#Set pandas to display all columns
# pd.set_option('display.max_columns',None)

#Adding the target column to dataframe
data_frame['label']=breast_cancer_dataset.target

#PRinting last 5 rows of dataframe
print(data_frame.tail())

#Number of rows and columns
print(data_frame.shape)

#Getting some info about data
data_frame.info()

#Checking for missing values
print(data_frame.isnull().sum())

#Statistical measures about data
print(data_frame.describe())

#Checking the distribution of taget variable
print(data_frame['label'].value_counts())

#Grouping data by label
print(data_frame.groupby('label').mean())

#Seperating features and target (X,Y)
X=data_frame.drop(columns='label',axis=1)
Y=data_frame['label']
# print(Y)

#Splitting data into Training and Testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
# print(X.shape, X_train.shape, X_test.shape) 

#Model Training(Logistic Regression)
model=LogisticRegression()
#Training logistic Regression model using Training Data
model.fit(X_train,Y_train) #Model Training
 
#Model Evaluation
#Accuracy Score on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy on training data= ',training_data_accuracy)

#Acuuracy Score on Test Data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy on testing data= ',testing_data_accuracy)

#Building a Predictive System
input_data=(12.46,24.04,83.97,475.9,0.1186,0.2396,0.2273,0.08543	,0.203	,0.08243	,0.2976	,1.599,	2.039,	23.94,	0.007149,	0.07217,	0.07743,	0.01432,	0.01789,	0.01008,	15.09,	40.68,	97.65,	711.4,	0.1853,	1.058,	1.105,	0.221,	0.4366,	0.2075,
)

#Change input data to numpy array for calculations
input_data_as_numpy_array=np.asarray(input_data)

#Reshape the numpy array as we are predecting for one data point
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("You are Cancer!")
else:
    print("You are not Cancer!!")