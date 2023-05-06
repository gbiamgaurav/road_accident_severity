import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils.utils import save_object, evaluate_models

from src.exception import CustomException
from src.logger import logging

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # We will choose the top 3 models found in EDA

            models = {
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "ExtratreeClassifier": ExtraTreesClassifier()
            }
            
            params ={"GradientBoosting":{
                    'loss':['exponential', 'log_loss', 'deviance'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]},
                
                "ExtraTreesClassifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8,16,32,64,128,256]
                }}
            
            
            model_report: dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               models=models, params=params)
            
            # To get best model score from dict
            
            best_model_score = max(sorted(model_report.values()))
            
            # To get the best model name from dict
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            score_f1 = f1_score(y_test, predicted, average="weighted")
            
            return score_f1
        
        except Exception as e:
            raise CustomException(e, sys)