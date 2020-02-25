from sklearn import datasets
import pandas as pd

patients = datasets.load_diabetes()

#print(patients['DESCR'])
#print(patients['data'].shape)
#print(patients['target'].shape)

patiens_p = pd.DataFrame(patients.data, columns=[patients.feature_names])
patiens_p.head()

#print("First patient in database")
#print(patients['data'][0,:])

#print(patients['target'][1])

#print(patiens_p.describe())

import matplotlib.pyplot as plt
import seaborn as sb

#sb.pairplot(patiens_p, diag_kind="kde")
#plt.show()

#expamle of data standardisation
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler
#sclaed_data = scaler.fit_transform(patients['data'])

from sklearn.model_selection import train_test_split
patients_train_data, patients_test_data, \
patients_train_target, patients_test_target = \
train_test_split(patients['data'], patients['target'], test_size=0.1)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(patients_train_data, patients_train_target)

id = 1
linear_regression_prediction = linear_regression.predict(patients_test_data[id,:].reshape(1,-1))
print(linear_regression_prediction)
print(linear_regression.coef_)
