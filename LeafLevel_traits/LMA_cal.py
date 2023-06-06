"""
calculate LMA
Ting Zheng
tzheng39@wisc.edu
"""
import numpy as np
import pandas as pd

dir_in = r'D:\GoogleDrive\Projects_ongoing\shift\data\LMA'
df_area = pd.read_csv(f'{dir_in}/LMA_scans_area.csv')
df_mass = pd.read_csv(f'{dir_in}/Leaf_dryMass.csv')

df_mass['sample number'] = df_mass['Sample Number'].astype(int)

df_lma = pd.merge(df_area, df_mass.loc[:, ['Dry Weights','sample number']], how='left', on='sample number')

# replace 'no dry weight' with nan
df_lma.loc[df_lma['Dry Weights']=='no dry weight', 'Dry Weights'] = np.nan

df_lma['lma g/m2'] = df_lma['Dry Weights'].astype(float)/df_lma['Area(cm^2)'].astype(float) * 10000

df_lma.to_csv(f'{dir_in}/shift_LMA.csv')