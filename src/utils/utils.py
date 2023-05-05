
import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            
            if isinstance(model, dict):
                parameters = params[model_name]
                model = model['model']
            else:
                parameters = params[model_name]
            
            gs = GridSearchCV(model, parameters, cv=3)
            
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
                        
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_score = f1_score(y_train, y_train_pred, average="weighted")
            
            test_model_score = f1_score(y_test, y_test_pred, average="weighted")
            
            report[model_name] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

    
    
    
def load_object(file_path):
    try:
        
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException