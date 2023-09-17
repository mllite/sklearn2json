
from sklearn.datasets import load_iris
from sklearn.svm import SVC
 
clf = SVC(random_state=1789, probability = True)

iris = load_iris()

clf.fit(iris.data, iris.target)

import converters.svm_converter as cvt

lConverter = cvt.svm_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)
