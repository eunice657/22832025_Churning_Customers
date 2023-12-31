# -*- coding: utf-8 -*-
"""Assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d7-F-4wPxEO0yjtBXmanQR_mbmaNP9G4
"""

import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

cc_data=pd.read_csv('/content/drive/MyDrive/CustomerChurn_dataset.csv')

cc_data.info()

cc_data.dtypes

cc_data.shape

cc_data.head()

cc_data.drop(columns=['customerID'],inplace=True)

import numpy as np
cc_data['TotalCharges'].replace(' ', np.nan,inplace=True)

cc_data.info()

#changing the datatype of totalcharges in order to impute
cc_data['TotalCharges'] = cc_data['TotalCharges'].astype(float)
cc_data['TotalCharges'].fillna(value=cc_data['TotalCharges'].median(), inplace=True)  # Impute NaNs with median

cc_data.info()

"""# **Exploratory Data Analysis**"""

import seaborn as sns
import matplotlib.pyplot as plt
#

#code for boxplot
fig, ax  = plt.subplots(1, 3, figsize=(20, 6))
sns.boxplot(x='Churn', y='tenure', data=cc_data, ax = ax[0])
sns.boxplot(x='Churn', y='MonthlyCharges', data=cc_data,ax = ax[1])
sns.boxplot(x='Churn', y='TotalCharges', data=cc_data,ax = ax[2])

plt.title('Box Plot for our Numerical Features')
plt.show()

"""interpretation"""

#creating a barchart to analyse our categorical values
sns.countplot(x='Contract', hue='Churn', data=cc_data)
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.show()

#creating a barchart to analyse our categorical values
sns.countplot(x='gender', hue='Churn', data=cc_data)
plt.title('Churn by gender')
plt.xlabel('gender')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.show()

#creating a barchart to analyse our categorical values
sns.countplot(x='InternetService', hue='Churn', data=cc_data)
plt.title('Churn by InternetService')
plt.xlabel('InternetService')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.show()

#creating a barchart to analyse our categorical values
plot=sns.countplot(x='PaymentMethod', hue='Churn', data=cc_data)
plt.title('Churn by PaymentMethod')
plt.xlabel('PaymentMethod')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

#creating a barchart to analyse our categorical values
sns.countplot(x='StreamingMovies', hue='Churn', data=cc_data)
plt.title('Churn by StreamingMovies')
plt.xlabel('StreamingMovies')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.show()

# CountPlot for the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=cc_data)
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Count of Churn')
plt.show()

#Churn Comparison Across Payment Methods
plt.figure(figsize=(10, 6))
sns.countplot(x='PaymentMethod', hue='Churn', data=cc_data)
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.title('Churn Comparison Across Payment Methods')
plt.legend(title='Churn', loc='upper right')
plt.show()

"""It can be seen that , in electronic check have customers that are likely to churn compared to the other payment methods."""

#Churn Comparison Across InternetService
plt.figure(figsize=(10, 6))
sns.countplot(x='InternetService', hue='Churn', data=cc_data)
plt.xlabel('InternetService')
plt.ylabel('Count')
plt.title('Churn Comparison Across InternetService')
plt.legend(title='Churn', loc='upper right')
plt.show()

"""Customers that use fiber optic as their internet service are likely to churn compared to those that do not have and those that use dsl."""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Encoding to convert categorical data into numerical data
encoded_copy=cc_data.copy()
for col in cc_data.columns:
    if cc_data[col].dtype == 'object':  # Check if column has non-numeric data type (assuming categorical)
        encoder = LabelEncoder()
        cc_data[col] = encoder.fit_transform(cc_data[col])

encoded_copy

#Scaling the data
import pandas as pd
from sklearn.preprocessing import StandardScaler
X=cc_data.drop('Churn',axis=1)
y=cc_data['Churn']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# new DataFrame with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

data = pd.concat([X_scaled_df, y], axis=1)

data

#Feature Extraction to select relevant features
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

X=data.drop('Churn',axis=1)
y=data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
val_x,x_test,val_y,Y_test,=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

model = RandomForestClassifier()

model.fit(X, y)

feature_importances = model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

selected_features = feature_importance_df['Feature'].values[:9]
print(selected_features)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_scale = sc.fit_transform(encoded_copy[selected_features])
train_scale

import pickle
with open('my_scalar.pkl','wb')as file:
  pickle.dump(sc,file)

"""**Building the ANN Mdel using Functional API**"""

#importing necessary libraries
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score

#using functional api
input_layer = Input(shape=(X_train.shape[1],))

#Hidden layer
hidden_layer_1 = Dense(32, activation='relu')(input_layer)
hidden_layer_2 = Dense(16, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(8, activation='relu')(hidden_layer_2)

#)Outputvlayer
output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

hist=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(val_x, val_y))

loss,accuracy = model.evaluate(X_test,y_test,verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:4f}")

!pip install tensorflow scikeras scikit-learn

"""# Using GridSearchCv and Keras Classifier"""

from sklearn.model_selection import KFold,GridSearchCV
from keras import Input,Model
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier

def keras_model(hidden_units=32):
  input_layer = Input(shape=(X_train.shape[1],))
  hidden_layer_1 = Dense(hidden_units, activation='relu')(input_layer)
  hidden_layer_2 = Dense(16, activation='relu')(hidden_layer_1)
  hidden_layer_3 = Dense(8, activation='relu')(hidden_layer_2)
  output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

  ker_model = Model(inputs=input_layer, outputs=output_layer)
  ker_model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

  return ker_model


model = KerasClassifier(build_fn=keras_model,epochs=10,batch_size=32,hidden_units=32,verbose=True)
cv=KFold(n_splits=3,shuffle=True,random_state=42)

parameter_grid = {
    'hidden_units':[32,64,128],
    'epochs': [10,20],
    'batch_size': [16, 32,64],
    'optimizer':['sgd','rmsprop','adam'],
    }

gr_cv = GridSearchCV(estimator=model, param_grid=parameter_grid, cv=cv, scoring='accuracy')
grid_result = gr_cv.fit(X_train,y_train,epochs=10,validation_data=(val_x, val_y),verbose=True,callbacks=[hist])

# Model Evaluation
print("Best Parameters: ", grid_result.best_params_)
print("Best Score: ", grid_result.best_score_)

the_model=grid_result.best_estimator_

#Predicting the performance
y_pred = the_model.predict(X_test)

#Accuracy Score
Accuracy = accuracy_score(y_test, y_pred)
AucScore = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {Accuracy}")
print(f"AUC Score: {AucScore}")

"""Retraining using best Parameters"""

#the best hyperparameters found
best_hyperparams=grid_result.best_params_
#hiddenunits
best_hidden_units = best_hyperparams['hidden_units']
#optimizer
best_optimizer = best_hyperparams['optimizer']

#batchsize
best_batch_size = best_hyperparams['batch_size']


#We will redefine our functional API model with the best hyperparameters
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer = Dense(best_hidden_units, activation='relu')(input_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)

best_model = Model(inputs=input_layer, outputs=output_layer)
best_model.compile(optimizer=best_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Retrain the best model on the entire training dataset
best_model.fit(X_train, y_train, epochs=10, batch_size=best_batch_size, verbose=True)

"""Evaluation on Test Data"""

#test on best model
y_pred_1=best_model.predict(X_test)

y_pred_val_1=best_model.predict(val_x)

#addressing issue of continous and binary values
threshold = 0.5  #threshold
y_pred_class = (y_pred_val_1 > threshold).astype(int)

#AUC
#AUC Score
AucScore = roc_auc_score(y_test, y_pred_1)
print(f"Retested AUC Score: {AucScore}")

#Accuracy Score
Accuracy = accuracy_score(y_test,(y_pred_1 > threshold).astype(int))
print(f"Retested Accuracy: {Accuracy}")

#Validation evaluation
val_acc=accuracy_score(val_y,y_pred_class )
print(f"Retested Validation Accuracy: {val_acc}")

#Loss
evaluation = best_model.evaluate(X_test, y_test)
loss = evaluation[0]
print(f"Loss: {loss:.4f}")

"""# Save Model"""

best_model.save('deployment.h5')