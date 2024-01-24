
from . import generic_converter as conv
import sklearn
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class random_forest_converter(conv.json_converter):
    def __init__(self):
        conv.json_converter.__init__(self);

    def get_model_options_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lOptions = ['n_estimators', 'ccp_alpha', 'criterion', 'max_depth', 'max_features',
                    'max_leaf_nodes', 'min_impurity_decrease', 'min_samples_leaf', 'min_samples_split',
                    'min_weight_fraction_leaf', 'random_state', 'n_jobs', 'max_samples', 'class_weight']
        for opt in lOptions:
            lDict[opt] = lDict1[opt]
        return lDict

    def get_sub_model_options_as_dict(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lOptions = ['ccp_alpha', 'criterion', 'max_depth', 'max_features',
                    'max_leaf_nodes', 'min_impurity_decrease', 'min_samples_leaf', 'min_samples_split',
                    'min_weight_fraction_leaf', 'random_state', 'splitter']
        for opt in lOptions:
            lDict[opt] = lDict1[opt]
        return lDict

    

    def get_tree_as_dict(self, clf):
        tree = clf.tree_
        # print(clf.__dict__)
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_sub_model_options_as_dict(clf)
        if(lDict1.get("classes_") is not None):
            lDict["classes"] = list(lDict1["classes_"])
        lTreeDict = {}
        node_count = len(tree.n_node_samples)
        lTreeDict["node_count"] = node_count
        lTreeDict["max_depth"] = tree.max_depth
        lTreeDict["features"] = lDict1['n_features_in_']
        lTreeDict["outputs"] = lDict1['n_outputs_']
        nodes = {}
        P = int(np.log(node_count) / np.log(10) + 1)
        for node_id in range(len(tree.n_node_samples)):
            node_id_str = ('0'*P + str(node_id))[-P:]
            # print("NODE_ID_STR ", P, node_id, node_id_str)
            normalized_value = tree.value[node_id][0]
            if(normalized_value.shape[0] > 1):
                normalized_value = normalized_value / np.sum(normalized_value)
            feature = tree.feature[node_id]
            (left, right, threshold) = (tree.children_left[node_id], tree.children_right[node_id], tree.threshold[node_id])
            if(feature == -2):
                feature = None
                (left, right, threshold) = (None, None, None)
            nodes["node_" + node_id_str ] = {
                "left" : left,
                "right" : right,
                "feature" : feature,
                "threshold" : threshold,
                "impurity" : tree.impurity[node_id],
                "n_samples" : tree.n_node_samples[node_id],
                "w_samples" : tree.weighted_n_node_samples[node_id],
                "value" : list(normalized_value)
            }
        lTreeDict["nodes"] = nodes
        lDict["tree"] = lTreeDict
        return lDict
        
    def convert_classifier(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        # print(lDict1)
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        if(lDict1.get("classes_") is not None):
            lDict["classes"] = list(lDict1["classes_"])
        lDict["forest"] = {}
        lDict["forest"]["Trees"] = clf.n_estimators
        P = int(np.log(clf.n_estimators) / np.log(10) + 1)
        for t in range(clf.n_estimators):
            node_id_str = ('0'*P + str(t))[-P:]
            lDict["forest"]["RF_Tree_" + node_id_str] = self.get_tree_as_dict(clf.estimators_[t])
        return lDict

    def convert_regressor(self, clf):
        lDict = {}
        lDict1 = clf.__dict__
        lDict["metadata"] = self.get_metadata(clf)
        lDict["options"] = self.get_model_options_as_dict(clf)
        lDict["forest"] = {}
        lDict["forest"]["Trees"] = clf.n_estimators
        P = int(np.log(clf.n_estimators) / np.log(10) + 1)
        for t in range(clf.n_estimators):
            node_id_str = ('0'*P + str(t))[-P:]
            lDict["forest"]["RF_Tree_" + node_id_str] = self.get_tree_as_dict(clf.estimators_[t])
        return lDict

    
    def convert_model(self, clf):
        if(clf.__class__ == sklearn.ensemble._forest.RandomForestClassifier):
            return self.convert_classifier(clf)
        if(clf.__class__ == sklearn.ensemble._forest.RandomForestRegressor):
            return self.convert_regressor(clf)
        return None
    
