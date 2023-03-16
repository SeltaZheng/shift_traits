"""
This script is used to match the spectra with species/plot meta
"""

import pandas as pd

dir_meta = r'G:\My Drive\Projects_ongoing\shift\data\meta'
meta = pd.read_csv(f'{dir_meta}/sample_list_10252022_full.csv', encoding='unicode_escape')

# create new column for unique ID: species+date+plot
meta['UID'] = meta['Species or type'] + '_' + meta['Sample Date'] + '_' + meta['Plot Name']

meta = meta[meta['Sample Taken?'] == 'Yes']

meta = meta.drop_duplicates(subset=['UID'], keep='first')

meta.to_csv(f'{dir_meta}/sample_list_simplified.csv', index=False)

# get unique type
stype = meta['Species or type'].unique()
df_sp = pd.DataFrame(data=stype, columns=['species'])
df_sp.to_csv(f'{dir_meta}/species.csv', index=False)