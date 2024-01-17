
import sklearn
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import sklearn2json.converters.decison_tree_converter as tree_cvt
import sklearn2json.converters.ridge_converter as ridge_cvt
import sklearn2json.converters.svm_converter as svm_cvt
import sklearn2json.converters.mlp_converter as mlp_cvt
import sklearn2json.converters.xgb_converter as xgb_cvt
import sklearn2json.converters.random_forest_converter as rf_cvt

def convert_model(clf):
    print("CONVERT_MODEL ", clf.__class__)    
    if(clf.__class__ == sklearn.tree._classes.DecisionTreeClassifier):
        lConverter = tree_cvt.decision_tree_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == sklearn.tree._classes.DecisionTreeRegressor):
        lConverter = tree_cvt.decision_tree_converter()
        return lConverter.convert_regressor(clf)
    if(clf.__class__ in [sklearn.svm._classes.SVC, sklearn.svm._classes.NuSVC]):
        lConverter = svm_cvt.svm_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ in [sklearn.svm._classes.SVR, sklearn.svm._classes.NuSVR]):
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
    if(clf.__class__ == xgb.XGBClassifier):
        lConverter = xgb_cvt.xgb_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == xgb.XGBRegressor):
        lConverter = xgb_cvt.xgb_converter()
        return lConverter.convert_regressor(clf)
    if(clf.__class__ == sklearn.ensemble.RandomForestClassifier):
        lConverter = rf_cvt.random_forest_converter()
        return lConverter.convert_classifier(clf)
    if(clf.__class__ == sklearn.ensemble.RandomForestRegressor):
        lConverter = rf_cvt.random_forest_converter()
        return lConverter.convert_regressor(clf)
    
    print("WARNING_CANNOT_CONVERT_MODEL ", clf.__class__)    
    return None
