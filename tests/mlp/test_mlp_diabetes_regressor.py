from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor

X, y = load_diabetes(return_X_y=True)


from sklearn.datasets import load_iris
 
clf = MLPRegressor(random_state=1789)

clf.fit(X, y)

print(clf.__class__)
print(clf.__dict__)

import converters.mlp_converter as cvt

lConverter = cvt.mlp_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)

