from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
X, y = load_diabetes(return_X_y=True)


from sklearn.datasets import load_iris
 
clf = Ridge(random_state=1789)

clf.fit(X, y)

print(clf.__class__)
print(clf.__dict__)
print(clf.intercept_)

import converters.ridge_converter as cvt

lConverter = cvt.ridge_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)

