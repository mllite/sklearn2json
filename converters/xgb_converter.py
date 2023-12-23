
from . import generic_converter as conv
import xgboost as xgb
import sklearn
import numpy as np
import ast, json

class xgb_converter(conv.json_converter):
    def __init__(self):
        conv.json_converter.__init__(self);

    def get_model_options_as_dict(self, clf):
        lDict = {}
        booster = clf.get_booster()
        config = json.loads(booster.save_config())
        print(json.dumps(config, sort_keys= True, indent = 4))
        lDict.update(config["learner"]["generic_param"])
        lDict.update(config["learner"]["learner_model_param"])
        lDict.update(config["learner"]["learner_train_param"])
        lDict.update(config["learner"]["objective"])
        lDict.update(config["learner"]["gradient_booster"]["gbtree_model_param"])
        lDict.update(config["learner"]["gradient_booster"]["gbtree_train_param"])
        lDict.update(config["learner"]["gradient_booster"]["tree_train_param"])
        lDict.update(config["learner"]["gradient_booster"]["updater"])
        return lDict

    def reformat_node(self, node_dict, parent_nidx = None):
        # print("XXXXXXXXXXXXXXXXXX", node_dict, parent_nidx)
        out_node_dict = {}
        out_node_dict["left"] = None
        out_node_dict["right"] = None
        out_node_dict["parent"] = parent_nidx
        out_node_dict["sindex"] = 0
        out_node_dict["svalue"] = 0
        out_node_dict["leaf_value"] = []
        nodeid = node_dict["nodeid"];
        
        if(node_dict.get("leaf") is not None):
            out_node_dict["leaf_value"] = [ node_dict["leaf"] ];
        else:
            out_node_dict["sindex"] = int(node_dict["split"][1:])
            out_node_dict["svalue"] = float(node_dict["split_condition"])
            out_node_dict["left"] = node_dict["yes"];
            out_node_dict["right"] = node_dict["no"];
            
        return (nodeid, out_node_dict)
    
    def linearize_tree(self, tree, parent = None):
        P = 2
        lDict = tree
        result = {}
        lDict1 = {}
        for x in lDict.keys():
            if(x != "children"):
                lDict1[x] = lDict[x]
        parent_nidx = lDict1["nodeid"]
        node_idx_str = ('0'*P + str(parent_nidx))[-P:]
        result["Node_" + node_idx_str] = self.reformat_node(lDict1, parent_nidx = parent)[1]
        if(lDict.get("children")):
            for v in lDict["children"]:
                lDict_c = self.linearize_tree(v, parent = parent_nidx)
                for(k1, v1) in lDict_c.items():
                    result[k1] = v1
        return result
    
    def get_booster_as_dict(self, clf):
        booster = clf.get_booster()
        config = json.loads(booster.save_config())
        print(config)
        intercept = config["learner"]["learner_model_param"]["base_score"]
        trees = booster.get_dump(dump_format="json")
        tree_count = len(trees)
        nodes = {}
        P = int(np.log(tree_count) / np.log(10) + 1)
        lDict = {}
        lDict["BaseSCore"] = intercept
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
        n_classes = int(lDict1["n_classes_"])
        lDict["classes"] = [x for x in range(n_classes)]
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
    
