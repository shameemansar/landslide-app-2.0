import pickle
import xgboost as xgb


class Model:
    def __init__(self,weight_path):
        self.weight_path = weight_path

        try:
            # Load from file
            self.model = xgb.XGBClassifier(reg_lambda=3,
                              reg_alpha= 1,
                              n_estimators= 400,
                              min_child_weight= 25,
                              max_depth= 10,
                              learning_rate= 0.1,
                              gamma= 1,
                              colsample_bytree= 1,
                              booster= 'gbtree')
            self.model.load_model(weight_path)
            print(f'model {weight_path} loaded')
        except:
            print(f'Unable to load model from {weight_path} Please check model path')

    
    def predict(self,data):
        out = None
        try:
            out = self.model.predict(data)
        except Exception as e:
            print(f'Error in prediction: {e}')
        
        return out

    def predict_prob(self,data):
        out = None
        try:
            out = self.model.predict_proba(data)
            out = out[:,1]
        except Exception as error:
            print('Error in prediction probability')
            print(error)
        return out

