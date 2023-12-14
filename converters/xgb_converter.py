
from . import generic_converter as conv
import xgboost as xgb
import sklearn
import numpy as np
import ast

class xgb_converter(conv.json_converter):
    def __init__(self):
        conv.json_converter.__init__(self);

    def get_model_options_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        print([x for x in lDict1.keys()])
        lOptions = [ "base_score", "early_stopping_rounds",
                     "eval_metric", "gamma", "grow_policy", "learning_rate",
                     "max_bin", "max_depth", "max_leaves", "min_child_weight",
                     "n_estimators", "objective", "reg_alpha", "reg_lambda",
                     "tree_method"]
        for opt in lOptions:
            lDict[opt] = lDict1[opt]
        return lDict

    def linearize_tree(self, tree):
        lDict = tree
        result = {}
        lDict1 = {}
        for x in lDict.keys():
            if(x != "children"):
                lDict1[x] = lDict[x]
        nidx = lDict1["nodeid"]
        result["Node_" + str(nidx)] = lDict1
        # print("XXXXXXXX", lDict1)
        # print("XXXXXXXX_1", result)
        if(lDict1.get("children")):
            for v in lDict["children"]:
                lDict_c = self.linearize_tree(v)
                for(k1, v1) in lDict_c.items():
                    nidx = v1["nodeid"]
                    result["Node_" + str(nidx)] = v1
        return result
    
    def get_booster_as_dict(self, clf):
        booster = clf.get_booster()
        trees = booster.get_dump(dump_format="json")
        tree_count = len(trees)
        nodes = {}
        P = int(np.log(tree_count) / np.log(10) + 1)
        lDict = {}
        for (idx, tree) in enumerate(trees):
            tree_idx_str = ('0'*P + str(idx))[-P:]
            tree_dict = ast.literal_eval(tree)
            lDict["Tree_" + tree_idx_str] = self.linearize_tree(tree_dict)
        return lDict
        
    def convert_classifier(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["classes"] = list(lDict1["classes_"])
        lDict["booster"] = self.get_booster_as_dict(clf)
        return lDict

    def convert_regressor(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["booster"] = self.get_booster_as_dict(clf)
        return lDict

    
    def convert_model(self, clf):
        if(clf.__class__ == xgb.XGBClassifier):
            return self.convert_classifier(clf)
        if(clf.__class__ == xgb.XGBRegressor):
            return self.convert_regressor(clf)
        return None
    
