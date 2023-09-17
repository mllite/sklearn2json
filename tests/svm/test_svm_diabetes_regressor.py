from sklearn.datasets import load_diabetes
from sklearn.svm import SVR

X, y = load_diabetes(return_X_y=True)


from sklearn.datasets import load_iris
 
clf = SVR()

clf.fit(X, y)

print(clf.__class__)
print(clf.__dict__)

import converters.svm_converter as cvt

lConverter = cvt.svm_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)

