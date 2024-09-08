#путь к модели предсказания зарплаты
path_model = 'model_1.cbm'

#путь к данным резюме
path_data ='vprod_test/TEST_SAL.csv'
#путь к создаваемому сабмишену
path_submit = 'submission_SAL_part.csv'


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

from tqdm.notebook import tqdm
tqdm.pandas()


def create_submission_part(test_df, name_of_predict_column, value):
    '''создание сабмишена под задачу'''
    submission = pd.DataFrame([])
    submission['id'] = test_df['id']
    submission[name_of_predict_column] = value
    return submission


df_TEST_SAL = pd.read_csv(path_data)

model = CatBoostRegressor()
model.load_model(path_model)

col = [ 'busy_type','education', 'education_speciality', 'regionName', 'company_business_size',
       'required_experience',  'schedule_type', 'source_type', 'professionalSphereName', 'federalDistrictCode']

cat_features = ['busy_type','education', 'education_speciality', 'regionName', 'company_business_size',
                'schedule_type', 'source_type', 'professionalSphereName']


X_test=df_TEST_SAL[col]
X_test=X_test.fillna(0)

test_pool = Pool(X_test, cat_features=cat_features)

test_predict = model.predict(test_pool)

submission_SAL_part = create_submission_part(df_TEST_SAL, 'salary', test_predict.round(1))

submission_SAL_part.to_csv(path_submit, index = False)
