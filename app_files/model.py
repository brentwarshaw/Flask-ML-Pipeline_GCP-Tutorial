# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('50_Startup.csv')

#Importing data set
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

###### START CODE HERE##############################
#define ANY regression Model here
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(max_depth=5)
reg.fit(X,y)
y_pred=reg.predict(X)
print(reg.score)
#######################################END here


#Save the model
pickle.dump(reg,open('treemodel.pkl','wb'))

# Loading the model to compare results
model=pickle.load(open('treemodel.pkl','rb'))
x_test=np.array([[16000, 135000, 450000]])
print(x_test)
print(model.predict(x_test))
