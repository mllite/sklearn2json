
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

import converters.decison_tree_converter as tree_cvt
import converters.ridge_converter as ridge_cvt
import converters.svm_converter as svm_cvt
import converters.mlp_converter as mlp_cvt

def convert_model(clf):
    print("CONVERT_MODEL ", clf.__class__)    
    if(clf.__class__ == sklearn.tree._classes.DecisionTreeClassifier):
        lConverter = tree_cvt.decision_tree_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == sklearn.tree._classes.DecisionTreeRegressor):
        lConverter = tree_cvt.decision_tree_converter()
        return lConverter.convert_regressor(clf)
    if(clf.__class__ == sklearn.svm._classes.SVC):
        lConverter = svm_cvt.svm_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == sklearn.svm._classes.SVR):
        lConverter = svm_cvt.svm_converter()
        return lConverter.convert_regressor(clf)
    if(clf.__class__ == sklearn.linear_model._ridge.RidgeClassifier):
        lConverter = ridge_cvt.ridge_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == sklearn.linear_model._ridge.Ridge):
        lConverter = ridge_cvt.ridge_converter()
        return lConverter.convert_regressor(clf)
    if(clf.__class__ == sklearn.neural_network._multilayer_perceptron.MLPClassifier):
        lConverter = mlp_cvt.mlp_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == sklearn.neural_network._multilayer_perceptron.MLPRegressor):
        lConverter = mlp_cvt.mlp_converter()
        return lConverter.convert_regressor(clf)
    
    print("WARNING_CANNOT_CONVERT_MODEL ", clf.__class__)    
    return None
