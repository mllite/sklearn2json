
from . import generic_converter as conv
import sklearn
from sklearn.svm import SVC

class svm_converter(conv.json_converter):
    def __init__(self):
        conv.json_converter.__init__(self);

    def get_model_options_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lOptions = ['kernel', 'degree', 'gamma', 'coef0', 'cache_size', 'epsilon', 'C']
        for opt in lOptions:
            lDict[opt] = lDict1[opt]
        return lDict

    def get_svm_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        L = clf.support_vectors_.shape[0]
        lDict["L"] = L
        lSVs = {}
        for sv_idx in range(L):
            lSVs["SV_" + str(sv_idx)] = list(clf.support_vectors_[sv_idx])
        lDict["SupportVectors"] = lSVs
        lCoefs = {}
        print(clf.dual_coef_.shape)
        for idx in range(clf.dual_coef_.shape[0]):
            lCoefs["SV_coef_" + str(idx)] = list(clf.dual_coef_[idx])
        lDict["SupportVectorsCoefs"] = lCoefs
        lDict["rho"] = list(-clf.intercept_)
        lDict["nSV"] = list(clf._n_support)
        lDict["probA"] = list(clf._probA)
        lDict["probB"] = list(clf._probB)
        lDict["n_iter"] = list(clf._num_iter)
        lDict["sv_ind"] = list(clf.support_)
        return lDict
        
    def convert_classifier(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["classes"] = list(clf.classes_)
        lDict["svm_model"] = self.get_svm_as_dict(clf)
        return lDict

    def convert_regressor(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["svm_model"] = self.get_svm_as_dict(clf)
        return lDict

    def convert_model(self, clf):
        print("CONVERT_SVM_MODEL ", clf.__class__)
        if(clf.__class__ == sklearn.svm._classes.SVC):
            return self.convert_classifier(clf)
        if(clf.__class__ == sklearn.svm._classes.SVR):
            return self.convert_regressor(clf)
        return None
    
