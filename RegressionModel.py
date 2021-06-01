import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split #to split data
from sklearn.linear_model import LinearRegression #to create the model
from sklearn.metrics import mean_squared_error, r2_score #to obtain some numbers

# df = pd.read_excel(r"./A01769659_EtiquetasNutrimentales.xlsx") #pip install openpyxl
df = pd.read_csv("./A01769659_EtiquetasNutrimentales.csv")

columns = ['Calorias (kcal)','Carbohidratos (g)','Lípidos (g)','Proteína (g)','Sodio (mg)']

kcals = df['Calorias (kcal)'].tolist()

x=df[["Carbohidratos (g)","Lípidos (g)","Proteína (g)","Sodio (mg)"]]
y=df[["Calorias (kcal)"]]


np.random.seed(2)
model = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.6)
train_errors, val_errors = [], []

for m in range(1, len(x_train)):
    model.fit(x_train[:m],y_train[:m])
    y_train_predict = model.predict(x_train[:m])
    y_val_predict = model.predict(x_test)
    train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
    val_errors.append(mean_squared_error(y_test,y_val_predict))


plt.plot(np.sqrt(train_errors),"r-+",label="training set")
plt.plot(np.sqrt(val_errors),"b-",label="validation set")
plt.legend(loc='best', frameon=False)
plt.ylabel('RMSE')
plt.xlabel('training set size')
plt.title   ('RMSE')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = len(y)-232)

print('Elements in training sample: %d'%len(y_train))
print('Elements in test     sample: %d'%len(y_test))

print('\nBuilding linear regression model\n')

#training model
print('\ntraining model ...\n')
model.fit(x_train,y_train)

#printing model
print('Coefficients: \n', model.coef_)
print('intercept: \n',model.intercept_)
print('\nModel:\nY = %f'%model.intercept_)
i=1
for par in model.coef_[0]:
    print(' %f x{%d}'%(1*par,i))
    i += 1

#printing x_test predictions
model_pred = model.predict(x_test)

#stats
print('\nMean squared error: %.2f'%mean_squared_error(y_test, model_pred))
print('Coefficient of determination: %.2f'%r2_score(y_test,model_pred))


#getting residuals and validations
stdev = df['Calorias (kcal)'].std()
prediction = model.predict(x)
prediction = prediction.flatten()

residuals = kcals - prediction
stdres = residuals/stdev

print("\nSuma de los residuos:", sum(residuals))
if abs(sum(residuals)) < 0.0001:
    print("Validación exitosa, la suma de los residuos es cercana a 0")
else:
    print("Validación fallida, la suma de los residuos no es cercana a 0")


figure2, axs = plt.subplots(2,2)

resdata = [['Residuals', 'Homogeneity of variance'], ['Test of Independence','Residuals Hist-Norm']]

for i in range(len(axs)):
    for j in range(len(axs[i])):
        axs[i][j].grid()
        axs[i][j].set_title(resdata[i][j])

axs[0][0].scatter(prediction,residuals)
axs[0][0].set_xlabel("Calorias (kcal)")
axs[0][0].set_ylabel("Calorias (kcal)")
axs[0][1].scatter(prediction,stdres)
axs[0][1].set_xlabel("Calorias (kcal)")
axs[0][1].set_ylabel("Sigmas")
axs[1][0].plot(range(1,len(residuals)+1),residuals)
axs[1][0].set_ylabel("Calorias (kcal)")

n_equal_bins = 30
bin_edges = np.linspace(start=min(residuals), stop=max(residuals), num=n_equal_bins + 1, endpoint=True)
axs[1][1].hist(residuals, bins=bin_edges, density=True)
mean,std=norm.fit(residuals)
X = np.linspace(min(residuals), max(residuals), 100)
Y = norm.pdf(X, mean, std)
axs[1][1].plot(X, Y)
plt.tight_layout()
plt.show()