import pandas as pd
import numpy as np

from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('credit_clients.xls')
#print(df.head())

data = df.iloc[1:,0:-1]
target = df.iloc[1:,-1]
#print(data.head())
#print(target.head())
data_np = np.array(data, dtype=np.int16)
target_np = np.array(target, dtype=np.int16)

#print(type(data_np))
#print(type(target_np))

#print(data_np.shape)
#print(target_np.shape)
from sklearn.preprocessing import StandardScaler
StandardScaler().fit(data_np)

from sklearn.model_selection import train_test_split

train_data, test_data, \
train_target, test_target = \
train_test_split(data_np, target_np, test_size= 0.2)

#print(train_data.shape)
#print(test_data.shape)
#print(train_target.shape)
#print(test_target.shape)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=500)
lr.fit(train_data,train_target)

id=1
prediction = lr.predict(test_data[id,:].reshape(1,-1))
#print("For index {0}: predict is - {1}".format(id, prediction))
#print("For index {0}: real value is - {1}".format(id, test_target[id]))

prediction_probability = lr.predict_proba(test_data[id,:].reshape(1,-1))
print(prediction_probability)

from sklearn.metrics import accuracy_score
acc = accuracy_score(test_target, lr.predict(test_data))
print("Model accuracy is {0:0.2f}".format(acc))

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_target, lr.predict(test_data))
print(conf_matrix)
