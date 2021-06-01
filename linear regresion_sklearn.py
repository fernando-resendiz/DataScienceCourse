import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #to split data
from sklearn.linear_model import LinearRegression #to create the model
from sklearn.metrics import mean_squared_error, r2_score #to obtain some numbers

#this is the ramdom seed to match the results, will discuss the impact of this
#choice later
np.random.seed(0)

dt = pd.read_excel(r"./A01769659_EtiquetasNutrimentales.xlsx") #pip install openpyxl

x=dt[["Carbohidratos (g)","Lípidos (g)","Proteína (g)","Sodio (mg)"]]
y=dt[["Calorias (kcal)"]]

#make sub-samples of traning and test data using a proportion of 1/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3)
print('Elements in training sample: %d'%len(y_train))
print('Elements in test     sample: %d'%len(y_test))

#creating a model
print('\nBuilding linear regression model\n')
model = LinearRegression()

#training model
print('\ntraining model ...\n')
model.fit(x_train,y_train)

#print the coefficients
print('Coefficients: \n', model.coef_)

#print intercept
print('intercept: \n',model.intercept_)

#print model
print('\nModel:\nY(x) = %f'%model.intercept_)
i=1
for par in model.coef_[0]:
    print(' %f x^{%d}'%(1*par,i))
    i += 1
    
#getting predictions from the model, note that for this we use the test sample,
#not the training sample
model_pred = model.predict(x_test)
npred = 5
print('\nshowing first %d predictions:\n'%npred)
print(model_pred[:npred])

#print mean squared error
print('\nMean squared error: %.2f'%mean_squared_error(y_test, model_pred))
#print the determination coefficient
print('Coefficient of determination: %.2f'%r2_score(y_test,model_pred))

#let's plot the data, remember that model.predict( ... is just to apply the model
#to any set of input data, it does not mean that we are training the dat again



