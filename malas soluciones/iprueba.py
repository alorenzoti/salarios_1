'''pip install numpy
pip install pandas
pip install seaborn
pip install sklearn3
'''


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler   

salaries = pd.read_csv('C:\ironhack\salarios_1\data\salaries_data.csv')

salaries.head()

test = pd.read_csv('C:/ironhack/salarios_1/data/testeo.csv')

test.head()

len(salaries.columns)==len(test.columns)

salaries.drop(columns=salaries[['salary', 'salary_currency']], axis=1, inplace=True)

len(salaries.columns)

salaries.info()

salaries_d =pd.get_dummies(salaries, columns=['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size'], drop_first=True)

data_num=pd.DataFrame(StandardScaler().fit_transform(salaries_d._get_numeric_data()),  # standardize numeric columns
                      columns=salaries_d._get_numeric_data().columns)

data_obj=salaries_d.select_dtypes(include='object')  # get categoric columns


data=pd.concat([data_num, data_obj], axis=1)   # concatenate both dataframes

test.columns

len(test.columns)

test_d = pd.get_dummies(test, columns=['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size'], drop_first=True)

data_num=pd.DataFrame(StandardScaler().fit_transform(test_d._get_numeric_data()),  # standardize numeric columns
                      columns=test_d._get_numeric_data().columns)

data_obj=test_d.select_dtypes(include='object')  # get categoric columns


data2=pd.concat([data_num, data_obj], axis=1)   # concatenate both dataframes

test_d

from pycaret.regression import *
s= setup(data, target = 'salary_in_usd')

best = compare_models(sort = 'RMSE')

evaluate_model(best)

predictions = predict_model(best,data = data2)

predictions

solution1 = predictions.prediction_label

solution1 = pd.DataFrame(predictions.prediction_label)
solution1

solution1.reset_index(drop=True, inplace=True)

solution1