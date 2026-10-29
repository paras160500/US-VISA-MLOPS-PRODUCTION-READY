from us_visa.logger import logging 
from us_visa.exception import USvisaException
import sys 


logging.info("This is the sample information...")

try:
    a = 2 / 0
except Exception as e:
    raise USvisaException(e , sys)