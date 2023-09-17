from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
X, y = load_diabetes(return_X_y=True)


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
 
clf = DecisionTreeRegressor(random_state=1789)

clf.fit(X, y)

import converters.decison_tree_converter as cvt

lConverter = cvt.decision_tree_converter()
lJSON = lConverter.convert_model(clf)

print(lJSON)

