import sys 

from us_visa.exception import USvisaException
from us_visa.logger import logging
from dotenv import load_dotenv

import os 
from us_visa.constants import DATABASE_NAME , MONGODB_URL_KEY
import pymongo
import certifi

load_dotenv()

ca = certifi.where()


class MongoDBClient:
    """
        Class Name : export_data_into_feture_Store
        Description : This method exports the dataframe from mongodb feature store as dict
        Output : connection to mongodb database
        On Failure : raise an exception
    """
    client = None 

    def __init__(self , database_name = DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                mongo_db_url = MONGODB_URL_KEY
                if mongo_db_url is None:
                    raise Exception(f"Environment key : {MONGODB_URL_KEY} is not set")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url , tlsCAFile = ca)
            self.client = MongoDBClient.client 
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successfull.")
        except Exception as e:
            print("ERror :- " , e)
            raise USvisaException(e , sys)
