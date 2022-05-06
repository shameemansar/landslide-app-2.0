import pandas as pd

def get_zone(prob):
    """
    get zone
    """
    zone = None
    if prob <= 0.25:
        zone = 'very low'
    elif prob > 0.25 and prob <= 0.5:
        zone = 'low'
    elif prob > 0.5 and prob <= 0.75:
        zone = 'high'
    else: 
        zone = 'very high'
    return zone

def fetch_panchayath(df,panchayath='Marayoor'):
    """
    fetch panchayath dataframe
    """
    df = df[df['PANCHAYATH_NAME'] == panchayath]
    if df.shape[0] >= 100:
        df = df.head(100)
    return df

def process_pipleine(df):
    """
    crate dataframa after one hot encoding
    
    """

    
    df = df.reset_index(drop=True)
    # one hot encoder
    known_lithology = ['Charnakite', 'Hornblende Gneiss', 'Pink Granite Gneiss',
        'Garnet-biotite Gneiss', 'Granite']


    known_lulc11_12 = ['Evergreen / Semi Evergreen Forest - Closed', 'Agriculture Plantation',
        'Scrub Forest', 'Forest Plantation',
        'Grassland - Temperate / Sub Tropical', 'Barren Rocky',
        'Deciduous Forest - Open', 'Rural', 'Scrubland - Closed',
        'Deciduous Forest - Closed', 'Reservoir - Permanent',
        'Evergreen / Semi Evergreen Forest - Open', 'Cropped in 2 Seasons',
        'Scrubland - Open', 'River - Perennial', 'Fallow Land',
        'Reservoir - Seasonal']

    known_lulc15_16 = ['Evergreen / Semi Evergreen Forest - Closed', 'Agriculture Plantation',
        'Forest Plantation', 'Barren Rocky', 'Deciduous Forest - Open', 'Rural',
        'Deciduous Forest - Closed', 'Cropped in 2 Seasons',
        'Grassland - Temperate / Sub Tropical',
        'Evergreen / Semi Evergreen Forest - Open', 'Reservoir - Permanent',
        'Scrubland - Closed', 'Scrubland - Open', 'Scrub Forest',
        'River - Perennial', 'Fallow Land', 'Reservoir - Seasonal',
        'Crop - Rabi']

    known_soil_depth = ['Very deep', 'Waterbody/Tank', 'Deep']
    known_soil_texture = ['Clay', 'Waterbody/Tank', 'Gravelly Loam', 'Loam', 'Gravelly Clay']
    known_soil_types = [i for i in range(1,12)]



    train_litho = pd.Categorical(df['LITHOLOGY'].values, categories = known_lithology)
    train_litho = pd.get_dummies(train_litho)


    train_lulc11_12 = pd.Categorical(df['LULC11_12'].values, categories = known_lulc11_12)
    train_lulc11_12 = pd.get_dummies(train_lulc11_12)
    train_lulc11_12.columns = [f'{i}_LULC11_12' for i in train_lulc11_12.columns]
   

    train_lulc15_16 = pd.Categorical(df['LULC15_16'].values, categories = known_lulc15_16)
    train_lulc15_16 = pd.get_dummies(train_lulc15_16)
    train_lulc15_16.columns = [f'{i}_LULC15_16' for i in train_lulc15_16.columns]


    train_soil_depth = pd.Categorical(df['SOIL_DEPTH'].values, categories = known_soil_depth)
    train_soil_depth = pd.get_dummies(train_soil_depth)
    train_soil_depth.columns = [f'{i}_SOIL_DEPTH' for i in train_soil_depth.columns]
 

    train_soil_texture = pd.Categorical(df['SOIL_TEXTURE'].values, categories = known_soil_texture)
    train_soil_texture = pd.get_dummies(train_soil_texture)
    train_soil_texture.columns = [f'{i}_SOIL_TEXTURE' for i in train_soil_texture.columns]

    train_soil_types = pd.Categorical(df['SOIL_T'].values, categories = known_soil_types)
    train_soil_types = pd.get_dummies(train_soil_types)
    train_soil_types.columns = [f'soil_type_K{i}' for i in range(1,12)]
    
    train_cat = pd.concat([train_litho,train_lulc11_12,train_lulc15_16,train_soil_depth,train_soil_texture,train_soil_types],axis=1)
    
    num_cols = ['HAND', 'TPI', 'ASPEC',
       'DRAIN_DEN', 'NDBI', 'NDVI', 'SPI', 'TRI', 'TWI', 'ELEVATION', 'PLANCU',
       'PROFCU', 'RRELIEF', 'GW_IDW', 'LINEAMENT_DEN', 'ROAD_DEN', 'SLOPE',
       'D_RAINFALL']
    df_ = pd.concat([df[num_cols],train_cat],axis=1)
    return df_
