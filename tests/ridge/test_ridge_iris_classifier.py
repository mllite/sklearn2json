
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
 
clf = RidgeClassifier(random_state=1789)

iris = load_iris()

clf.fit(iris.data, iris.target)

import converters.ridge_converter as cvt

lConverter = cvt.ridge_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)
