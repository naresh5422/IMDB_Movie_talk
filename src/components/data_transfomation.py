import sys
import os
import nltk
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download("stopwords")
from nltk.stem import PorterStemmer, WordNetLemmatizer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """ This function is responsible for Data Transformation"""
        try:
            text = ["review"]
            logging.info(f'review_data: {text}')
            spwd = set(stopwords.words('english'))

            # Using a variable for the top k features to be selected
            top_k_features=1000
            text_processor = Pipeline([
            ('count vectorizer',CountVectorizer(stop_words=spwd,lowercase=True)),
            ('chi2score',SelectKBest(chi2,k=top_k_features)),
            ('tf_transformer',TfidfTransformer(use_idf=True))])

            preprocessing = ColumnTransformer([('Text_pipeline', text_processor, text)])

            label = ["sentiment"]
            logging.info(f'sentiments: {label}')

            return preprocessing    
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path, sep=',')
            test_df = pd.read_csv(test_path, sep=',')

            logging.info("Read train and test data has completed")

            logging.info("Obtaining preprocessing Object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            num_cols = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing object has saved")

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessor_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)