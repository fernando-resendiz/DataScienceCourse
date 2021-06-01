import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm
# from DataInfo import getDataInfo #modulo personal

#df = pd.read_excel(r"./A01769659_EtiquetasNutrimentales.xlsx") #pip install openpyxl
df = pd.read_csv("./A01769659_EtiquetasNutrimentales.csv")

columns = ['Calorias (kcal)','Carbohidratos (g)','Lípidos (g)','Proteína (g)','Sodio (mg)']


# getDataInfo(df,columns)
# print('\n\n',df.describe(),'\n\n')

kcals = df['Calorias (kcal)'].tolist()
carbs = df['Carbohidratos (g)'].tolist()
lipids = df['Lípidos (g)'].tolist()
prots = df['Proteína (g)'].tolist()
sod = df['Sodio (mg)'].tolist()


model = smf.ols('kcals ~ carbs + lipids + prots + sod', data=df).fit()

print(model.summary())
print('---------------------------------------')

coefs = model.params

prediction = model.predict()
residuals = kcals - prediction
stdev = df['Calorias (kcal)'].std()
stdres = residuals/stdev

print("\nSuma de los residuos:", sum(residuals))

if sum(residuals) < 0.0001:
    print("Validación exitosa, la suma de los residuos es cercana a 0")
else:
    print("Validación fallida, la suma de los residuos no es cercana a 0")

print("\n\n")
print("Calorias:",kcals[:5])
print("Prediction:",prediction[:5])
print("Residuals:",residuals[:5])

figure1 = plt.figure()
n_equal_bins = 10
for i in range(1,6):
    axes = figure1.add_subplot(2,3,i)
    a = df[columns[i-1]]
    bin_edges = np.linspace(start=a.min(), stop=a.max(), num=n_equal_bins + 1, endpoint=True)
    df.hist(column=columns[i-1], ax=axes, bins=bin_edges, density=True)
    mean,std=norm.fit(a.tolist())
    X = np.linspace(a.min(), a.max(), 100)
    Y = norm.pdf(X, mean, std)
    plt.plot(X, Y)
    axes.set_title(columns[i-1])
    i += 1

figure2, axs = plt.subplots(2,2)

resdata = [['Residuals', 'Residuals/STD'], ['Residuals Pattern','Residuals Hist-Norm']]

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

n_equal_bins = 20
bin_edges = np.linspace(start=min(residuals), stop=max(residuals), num=n_equal_bins + 1, endpoint=True)
axs[1][1].hist(residuals, bins=bin_edges, density=True)
mean,std=norm.fit(residuals)
X = np.linspace(min(residuals), max(residuals), 100)
Y = norm.pdf(X, mean, std)
axs[1][1].plot(X, Y)
plt.tight_layout()
plt.show()