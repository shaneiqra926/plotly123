# Importing the libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model 


df = pd.read_csv('PakUrban.csv')

reg = linear_model.LinearRegression()

reg.fit(df[['Year']],df.Population)

print(reg.predict([[1968]]))

print(reg.coef_)

print(reg.intercept_)

print(1049113.46182432*1968+-2048128557.078829)

pickle.dump(reg, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2050]]))



