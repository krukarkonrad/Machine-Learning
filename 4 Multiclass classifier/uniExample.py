import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import confusion_matrix

multiple_classes_data = datasets.load_wine()

wine_train_data, wine_test_data, \
wine_train_target, wine_test_target = \
train_test_split(multiple_classes_data.data, multiple_classes_data.target, test_size=0.1)

#initiate classifier
multiclass_classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))

#fit classifier
multiclass_classifier.fit(wine_train_data, wine_train_target);

conf_matrix = confusion_matrix(wine_test_target, multiclass_classifier.predict(wine_test_data))
print("Confusion_matrix:")
print(conf_matrix)

#Recognizing hand-written digits with multiclass logistic regression (OvR strategy)

import matplotlib.pyplot as plt

digits = datasets.load_digits()
print("Examples in dataset: ", digits.data.shape[0])

id = 100
plt.title("Number: " + str(digits.target[id]))
plt.imshow(digits.images[id], cmap=plt.cm.gray_r, interpolation='nearest')

print("Picture vectorization:\n", digits.data[id])
#freeze script and give control to new window!
plt.show()

