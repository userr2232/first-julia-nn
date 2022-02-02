import subprocess
from sqlalchemy import create_engine
import pymysql
import pandas as pd

p = subprocess.Popen(["./cloud_sql_proxy", "-instances=deep-learning-308822:us-central1:hpo=tcp:3306"], 
                     stdout=subprocess.PIPE)

engine = create_engine('mysql+pymysql://userr2232@127.0.0.1/first_julia_nn')
df = pd.read_sql('SHOW TABLES', con=engine)

print(df)