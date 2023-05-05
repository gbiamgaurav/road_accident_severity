
import os 
import sys
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/rta_data.csv")
            df = df.drop(columns=['Time'], axis=1)
            logging.info("Read the dataset as dataframe")
            
            # renaming the columns
            df = df.rename(columns={
                'Day_of_week': 'weekday', 
                'Age_band_of_driver': 'driver_age', 
                'Sex_of_driver': 'driver_sex',
                'Educational_level': 'educational_level', 
                'Vehicle_driver_relation': 'driver_relation', 
                'Driving_experience': 'driving_exp',
                'Type_of_vehicle': 'vehicle_type', 
                'Owner_of_vehicle': 'vehicle_owner',
                'Service_year_of_vehicle': 'service_year',
                'Defect_of_vehicle': 'vehicle_defect', 
                'Area_accident_occured': 'accident_area', 
                'Lanes_or_Medians': 'lanes',
                'Road_allignment': 'road_alignment',
                'Types_of_Junction': 'junction_type',
                'Road_surface_type': 'road_type',
                'Road_surface_conditions': 'road_conditions', 
                'Light_conditions': 'light_condition', 
                'Weather_conditions': 'weather_condition',
                'Type_of_collision': 'collision_type', 
                'Number_of_vehicles_involved': 'num_of_vehicles',
                'Number_of_casualties': 'casualty', 
                'Vehicle_movement': 'vehicle_movement', 
                'Casualty_class': 'casualty_class',
                'Sex_of_casualty': 'casualty_sex', 
                'Age_band_of_casualty': 'casualty_age',
                'Casualty_severity': 'casualty_severity',
                'Work_of_casuality': 'casualty_work', 
                'Fitness_of_casuality': 'casualty_fitness', 
                'Pedestrian_movement': 'pedestrian_movement',
                'Cause_of_accident': 'accident_cause', 
                'Accident_severity': 'accident_severity'}
)
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=40)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data, *_ = obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr, test_arr, _, = data_transformation.initiate_data_transformation(train_data, test_data)
    
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
    