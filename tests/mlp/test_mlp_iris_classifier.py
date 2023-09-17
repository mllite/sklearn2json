
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
 
clf = MLPClassifier(random_state=1789)

iris = load_iris()

clf.fit(iris.data, iris.target)

import converters.mlp_converter as cvt

lConverter = cvt.mlp_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)
