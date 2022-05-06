import numpy as np 
import pandas as pd
import os 
from model import *
from flask import Flask, request, jsonify
import json
from utils import *

model_path = 'xgb_model.pkl'
db_path = 'db.csv'
MODEL = Model(model_path)
DB = pd.read_csv(db_path)





def predict_slide(df,rain_water_level,gwd_water_level):
    drop_cols = ['X','Y','PANCHAYATH_CODE','VILLAGE_BLK','Original Slide label', 'Static Predicted Probability',
       'Static predicted zone','PANCHAYATH_NAME']
    df_ = df.drop(columns=drop_cols)
    df_['D_RAINFALL'] = [int(rain_water_level) for i in range(df.shape[0])]
    df_['GW_IDW'] = [int(gwd_water_level) for i in range(df.shape[0])]
    df_ = process_pipleine(df_)
    df['Dynamic predicted probability'] = MODEL.predict_prob(df_)
    df['Dynamic predicted zone'] = df['Dynamic predicted probability'].apply(lambda x: get_zone(x))
    
    df['S_RAINFALL'] = df_['D_RAINFALL']
    df['S_GW_IDW'] = df_['GW_IDW']

    return df

app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)
    panchayath_name = data['panchayath_name']
    rain_water_level = data['rain_water_level']
    gwd_water_level = data['ground_water_level']

    df = fetch_panchayath(DB,panchayath=panchayath_name)
    out = predict_slide(df,rain_water_level,gwd_water_level)
    out_dict = out.to_dict('list')
    return jsonify(success="true", data=out_dict)


if __name__== "__main__":
   app.run(port=5000,debug=False)


    # data = {'panchayath_name':'Marayoor','rain_water_level':2,'ground_water_level':1}

    # panchayath_name = data['panchayath_name']
    # rain_water_level = data['rain_water_level']
    # gwd_water_level = data['ground_water_level']
    
    # df = fetch_panchayath(DB,panchayath=panchayath_name)
    # out = predict(df,rain_water_level,gwd_water_level)
    
    # print(out)


     
    









