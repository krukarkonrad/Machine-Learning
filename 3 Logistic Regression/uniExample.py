import numpy as np
from sklearn import datasets
patients = datasets.load_breast_cancer()
#print(patients.DESCR)
print(patients.data.shape)
print(patients.target.shape)
#print("First patient in database")
#print(patients['data'][0,:])

from sklearn.model_selection import train_test_split

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler
#sclaed_data = scaler.fit_transform(patients['data'])
#sclaed_target = scaler.fit_transform(patients['target'])

patients_train_data, patients_test_data, \
patients_train_target, patients_test_target = \
train_test_split(patients['data'], patients['target'], test_size = 0.1)

print("Training dataset:")
print("patients_train_data", patients_train_data.shape)
print("patients_train_target", patients_train_target.shape)

print("Testing dataset:")
print("patients_test_data", patients_test_data.shape)
print("patients_test_target", patients_test_target.shape)

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(max_iter=4000)
logistic_regression.fit(patients_train_data, patients_train_target)

id=6
prediction = logistic_regression.predict(patients_test_data[id,:].reshape(1,-1))
print("Model predicted for patient {0} value {1}".format(id, prediction))

print("Real value for patient \"{0}\" is {1}".format(id, patients_test_target[id]))

prediction_probability = logistic_regression.predict_proba(patients_test_data[id,:].reshape(1,-1))
print(prediction_probability)

from sklearn.metrics import accuracy_score
acc = accuracy_score(patients_test_target, logistic_regression.predict(patients_test_data))
print("Model accuracy is {0:0.2f}".format(acc))

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(patients_test_target, logistic_regression.predict(patients_test_data))
print(conf_matrix)