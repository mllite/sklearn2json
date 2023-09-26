
from . import generic_converter as conv
import sklearn
from sklearn.svm import SVC
import numpy as np
           
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
        P = int(np.log(L) / np.log(10) + 1)
        for sv_idx in range(L):
            sv_str = ('0'*P + str(sv_idx))[-P:]
            lSVs["SV_" + sv_str] = list(clf.support_vectors_[sv_idx])
        lDict["SupportVectors"] = lSVs
        lCoefs = {}
        print(clf.dual_coef_.shape)
        P = int(np.log(clf.dual_coef_.shape[0]) / np.log(10) + 1)
        for idx in range(clf.dual_coef_.shape[0]):
            coef_str = ('0'*P + str(idx))[-P:]
            lCoefs["SV_coef_" + coef_str] = list(clf.dual_coef_[idx])
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
    
