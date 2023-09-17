


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
 
clf = DecisionTreeClassifier(random_state=1789)
iris = load_iris()

clf.fit(iris.data, iris.target)

import converters.decison_tree_converter as cvt

lConverter = cvt.decision_tree_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)
