import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

classifier = svm.SVC(gamma=1, C=10000)

# print(len(digits.data))

x, y = digits.data[: -1], digits.target[: -1]
classifier.fit(x, y)

img = -138
number = digits.data[[img]]
actual_number = digits.target[img]
prediction = classifier.predict(number)
title = 'Prediction = ' + str(prediction) + ' Actual Number = ' + str(actual_number)

plt.title(str(title))
plt.imshow(digits.images[img], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()