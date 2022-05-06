import numpy as np 
import pandas as pd
import os 
from model import *
from utils import *
import streamlit as st


model_path = 'xgb_model.json'
db_path = 'db.csv'

MODEL_XGB = Model(model_path)


DB = pd.read_csv(db_path)


def predict_slide(df,rain_water_level,gwd_water_level):
    df_ = df.copy()
    drop_cols = ['X','Y','PANCHAYATH_CODE','VILLAGE_BLK','Original Slide label', 'Static Predicted Probability',
       'Static predicted zone','PANCHAYATH_NAME']
    order_cols_1 = ['X','Y','PANCHAYATH_NAME','D_RAINFALL',
                 'GW_IDW','Static Predicted Probability',
                 'Static predicted zone','Newly predicted probability','Newly predicted zone']
    df_ = df_.drop(columns=drop_cols)
    if rain_water_level is not None and gwd_water_level is None:
        df_['D_RAINFALL'] = [int(rain_water_level) for i in range(df_.shape[0])]
        df['D_RAINFALL_new'] = [int(rain_water_level) for i in range(df_.shape[0])]
        order_cols_1.insert(3,'D_RAINFALL_new')
        
    elif gwd_water_level is not None and rain_water_level is None:
        df_['GW_IDW'] = [int(gwd_water_level) for i in range(df_.shape[0])]
        df['GW_IDW_new'] = [int(gwd_water_level) for i in range(df_.shape[0])]
        order_cols_1.insert(3,'GW_IDW_new')

    elif gwd_water_level is not None and rain_water_level is not None:
        df_['D_RAINFALL'] = [int(rain_water_level) for i in range(df_.shape[0])]
        df['D_RAINFALL_new'] = [int(rain_water_level) for i in range(df_.shape[0])]
        order_cols_1.insert(3,'D_RAINFALL_new')
        df_['GW_IDW'] = [int(gwd_water_level) for i in range(df_.shape[0])]
        df['GW_IDW_new'] = [int(gwd_water_level) for i in range(df_.shape[0])]
        order_cols_1.insert(3,'GW_IDW_new')
    
    else:
        pass

    df_ = process_pipleine(df_)
    df['Newly predicted probability'] = MODEL_XGB.predict_prob(df_)
    df['Newly predicted zone'] = df['Newly predicted probability'].apply(lambda x: get_zone(x))


    # arranging columns    
    order_cols_2 = [c for c in df.columns if c not in order_cols_1]

    df = df[order_cols_1 + order_cols_2]

    return df





st.set_page_config(
          page_title="Land slide prediction (influence of rainfall and ground water level)",
          page_icon="🧊",
          layout="wide",
          initial_sidebar_state="expanded",
         )
st.title('Land Slide Prediction Idukki')
st.subheader('Find the influence of rainfall/ground water level on landslides in each Pachayath')

with st.form("form1", clear_on_submit= False):

   
   panchayath_name = st.selectbox("Enter Panchayath name",
                                  ('Adimali', 'Arakulam', 'Ayyappancoil', 'Baisonvalley',
                                 'Chakkupallam', 'Chinnakanal', 'Devikulam', 'Edamalakudy',
                                 'Elappara', 'Erattayar', 'Idukki-Kanjikuzhy', 'Kamakshy',
                                 'Kanchiyar', 'Kanthalloor', 'Karimannoor', 'Karunapuram',
                                 'Kattappana', 'Kokkayar', 'Konnathady', 'Kudayathoor', 'Kumily',
                                 'Mankulam', 'Marayoor', 'Mariyapuram', 'Munnar', 'Nedumkandam',
                                 'Pallivasal', 'Pampadumpara', 'Peermade', 'Peruvanthanam',
                                 'Rajakkad', 'Rajakumary', 'Santhanpara', 'Senapathy',
                                 'Thodupuzha Municipality', 'Udumbannoor', 'Vandanmedu',
                                 'Vannapuram', 'Vathikudy', 'Vattavada', 'Vazhathope',
                                 'Vellathooval', 'Velliyamattom'))
   st.write('You selected:', panchayath_name)

   rain_water_level = st.selectbox("Enter Rain water level(in mm)",
                                   ('No change','1(<8mm)','2(8-16mm)','3(16.1-24mm)','4(24.1-33mm)',
                                   '5(>32mm)'))
   st.write('You selected:', rain_water_level)
   rain_water_map = {'No change':None, '1(<8mm)':1,'2(8-16mm)':2,'3(16.1-24mm)':3,'4(24.1-33mm)':4,'5(>32mm)':5}
   rain_water_level = rain_water_map[rain_water_level]


   gwd_water_level = st.selectbox("Enter ground water level(in m)",
                                   ('No change','1(0-0.5m)','2(0.51-1.25m)','3(1.26-2m)','4(2-2.75m)',
                                   '5(2.75-3.5m)','6(3.6-4.25m)','7(4.25-5m)','8(5-5.75m)',
                                   '9(5.75-5.5m)','10(>5.5m)'))
   st.write('You selected:', gwd_water_level)
   gwd_water_map = {'No change': None,
                   '1(0-0.5m)':1,'2(0.51-1.25m)':2,'3(1.26-2m)':3,'4(2-2.75m)':4,'5(2.75-3.5m)':5,
                   '6(3.6-4.25m)':6,'7(4.26-5m)':7,'8(5.1-5.75m)':8,'9(5.76-6.5m)':9,'10(>6.5m)':10
                   }
   gwd_water_level = gwd_water_map[gwd_water_level]
   submit = st.form_submit_button("Predict")
   



   if submit:
       df = fetch_panchayath(DB,panchayath=panchayath_name)
       out_df = predict_slide(df,rain_water_level,gwd_water_level)
       out_df = out_df.reset_index(drop=True)

      #  df_map = out_df[['X','Y']].rename(columns={'X':'lat','Y':'lon'})
      #  st.map(df_map)
       st.text(' ')
       st.text(' ')
       st.subheader('Output:')
       with st.container():
           st.text(' ')
           st.text(f'Output dataframe showing probability of land slides in {panchayath_name} if rainfall/ground water level remains as above')
           st.dataframe(data=out_df, width=None, height=None)
       
       if rain_water_level is not None and gwd_water_level is None:
           filterd_cols = ['X','Y','D_RAINFALL','D_RAINFALL_new',
           'Original Slide label', 'Static Predicted Probability','Static predicted zone',
           'Newly predicted probability','Newly predicted zone']
       elif gwd_water_level is not None and rain_water_level is None:
           filterd_cols = ['X','Y','GW_IDW','GW_IDW_new',
           'Original Slide label', 'Static Predicted Probability','Static predicted zone',
           'Newly predicted probability','Newly predicted zone']
       
       elif gwd_water_level is not None and rain_water_level is not None:
           filterd_cols = ['X','Y','D_RAINFALL','D_RAINFALL_new','GW_IDW','GW_IDW_new',
           'Original Slide label', 'Static Predicted Probability','Static predicted zone',
           'Newly predicted probability','Newly predicted zone']

       else:
           filterd_cols = ['X','Y','D_RAINFALL','GW_IDW',
           'Original Slide label', 'Static Predicted Probability','Static predicted zone']
       
       st.text(' ')
       st.text(' ')
       out_df_2 = out_df[filterd_cols]
       with st.container():
           st.text(' ')
           st.text(f'Selected columns related to {panchayath_name} panchayath')
           st.dataframe(data=out_df_2, width=None, height=None)

        




