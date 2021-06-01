import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #to split data
from sklearn.linear_model import LinearRegression #to create the model
from sklearn.metrics import mean_squared_error, r2_score #to obtain some numbers

#this is the ramdom seed to match the results, will discuss the impact of this
#choice later
#np.random.seed(0)

#import data from a known dataset
dataset = pd.read_csv('Salary_Data.csv')

#indexin input data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#make sub-samples of traning and test data using a proportion of 1/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3)
print('Elements in training sample: %d'%len(x_train))
print('Elements in test     sample: %d'%len(x_test))

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
print('\nModel:\nY(x) = %fx + %f$'%(model.coef_,model.intercept_))

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

y_model = model.predict(x_train) # this is the model, and it will be unique

plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, y_model, color = "black")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, y_model, color = "blue")
plt.title("Salary vs Experience (Testing set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#do the results always look the same? remember the random seed?, try setting
#that to a fixed value and try again to train the model, a couple of times.
