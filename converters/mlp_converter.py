
from . import generic_converter as conv
import sklearn
from sklearn.neural_network import MLPClassifier
import numpy as np

class mlp_converter(conv.json_converter):
    def __init__(self):
        conv.json_converter.__init__(self);

    def get_model_options_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lOptions = ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size', 'learning_rate',
                    'learning_rate_init', 'power_t', 'max_iter', 'shuffle', 'random_state', 'tol', 'verbose',
                    'warm_start', 'momentum', 'nesterovs_momentum', 'early_stopping', 'validation_fraction',
                    'beta_1', 'beta_2', 'epsilon', 'n_iter_no_change', 'max_fun']
        for opt in lOptions:
            lDict[opt] = lDict1[opt]
        return lDict

    def get_mlp_as_dict(self, clf):
        lDict1 = clf.__dict__
        NL = len(lDict1['coefs_'])
        lSizes = [lDict1['coefs_'][0].shape[0]] + [lDict1['coefs_'][layer].shape[1] for layer in range(NL)]
        lLayers = {}
        lLayers["sizes"] = lSizes
        lLayers["Layer_0"] = { "name" : "Input_Layer", "NbInputs" : 0, "NbOutputs" : lSizes[0], "intercepts" : [ ] }
        P = int(np.log(NL) / np.log(10))
        for layer in range(1, NL + 1):
            layer_info = {}
            layer_str = ('0'*P + str(layer))[-P:]
            layer_info["name"] = "Hidden_Layer_" + layer_str
            if(layer == NL):
                layer_info["name"] = "Output_Layer"
            lCoefs = lDict1['coefs_'][layer - 1]
            layer_info["NbInputs"] = lCoefs.shape[0]
            layer_info["NbOutputs"] = lCoefs.shape[1]
            P1 = int(np.log(lCoefs.shape[0]) / np.log(10))
            for idx in range(lCoefs.shape[0]):
                idx_str = ('0'*P1 + str(idx))[-P1:]
                layer_info["coeffs_" + idx_str] = list(lCoefs[idx])
            layer_info["intercepts"] = list(lDict1['intercepts_'][layer - 1])
            lLayers["Layer_" + str(layer)] = layer_info
        return lLayers
        
    def convert_classifier(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["classes"] = list(clf.classes_)
        lDict["layers"] = self.get_mlp_as_dict(clf)
        return lDict

    def convert_regressor(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["layers"] = self.get_mlp_as_dict(clf)
        return lDict

    def convert_model(self, clf):
        print("CONVERT_MLP_MODEL ", clf.__class__)
        if(clf.__class__ == sklearn.neural_network._multilayer_perceptron.MLPClassifier):
            return self.convert_classifier(clf)
        if(clf.__class__ == sklearn.neural_network._multilayer_perceptron.MLPRegressor):
            return self.convert_regressor(clf)
        return None
    
