import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("Hiring.csv")

data['experience'].fillna(0,inplace = True)

def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2 ,'three': 3,'four':4,'five':5,'six': 6, 'seven': 7 ,'eight':8,'nine':9
                 ,'zero':0,0:0}
    return word_dict[word]

x = data.iloc[:,:3]

x['experience'] = x['experience'].apply(lambda a : convert_to_int(a))

y= data.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x,y)

pickle.dump(regressor,open('Salary_Prediction.pkl','wb'))

model = pickle.load(open('Salary_Prediction.pkl','rb'))

print(model.predict([[2,9,6]]))




