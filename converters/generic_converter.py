
import sklearn

class json_converter:

    def __init__(self):
        pass

    def get_full_class_name(self, clf):
        lClass = clf.__class__
        return lClass.__module__ + '.' + lClass.__name__
    
    def get_metadata(self, clf):
        lDict = {
            "model" : self.get_full_class_name(clf),
            "version" : sklearn.__version__
        }
        return lDict;
        
