
from . import generic_converter as conv
import sklearn
from sklearn.linear_model import RidgeClassifier

class ridge_converter(conv.json_converter):
    def __init__(self):
        conv.json_converter.__init__(self);

    def get_model_options_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lOptions = ['alpha', 'random_state']                    
        for opt in lOptions:
            lDict[opt] = lDict1[opt]
        return lDict

    def get_equations_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        equations = {}
        if(len(clf.coef_.shape) == 1):
            equations["coeffs"] = list(clf.coef_)
            equations["intercept"] = clf.intercept_
        else:
            for class_idx in range(clf.coef_.shape[0]):
                info = {
                    "coef" : list(clf.coef_[class_idx]),
                    "intercept" : clf.intercept_[class_idx]
                }
                equations["class_" + str(class_idx)] = info
        return equations
        
    def convert_classifier(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["classes"] = list(clf._label_binarizer.classes_)
        lDict["equations"] = self.get_equations_as_dict(clf)
        return lDict

    def convert_regressor(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["equations"] = self.get_equations_as_dict(clf)
        return lDict

    def convert_model(self, clf):
        print("CONVERT_RIDGE_MODEL ", clf.__class__)
        if(clf.__class__ == sklearn.linear_model._ridge.RidgeClassifier):
            return self.convert_classifier(clf)
        if(clf.__class__ == sklearn.linear_model._ridge.Ridge):
            return self.convert_regressor(clf)
        return None
    
