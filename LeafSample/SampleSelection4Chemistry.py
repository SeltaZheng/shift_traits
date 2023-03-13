"""
This script sort out field sample record and perform sample selection for chemistry analyses
"""
import pandas as pd
import numpy as np

dir_in = r'D:\GoogleDrive\Projects_ongoing\shift\data\meta'

df_sample = pd.read_csv(f'{dir_in}/processed/sample_list_simplified.csv')
df_inv = pd.read_csv(f'{dir_in}/processed/SHIFT_sample_inventory_20230308.csv')
# Clean the df_inv
df_inv = df_inv[df_inv['Sample Number'] != '131_flower']
df_inv = df_inv[df_inv['Sample Number'] != '161_S'].reset_index(drop=True)
# Step1: check if there are any missing samples:
## 1. between CA sample list and UWM sample inventery
n_sample = df_sample['Sample Number'].unique().astype(int)
n_inv = df_inv['Sample Number'].unique().astype(int)
n_com = set(n_sample).intersection(set(n_inv))
n_sample_only = set(n_sample).difference(n_com)
n_inv_only = set(n_inv).difference(n_com)

df_sample_only = df_sample[df_sample['Sample Number'].isin(n_sample_only)]
df_inv_only = df_inv[df_inv['Sample Number'].astype(int).isin(n_inv_only)]

df_sample_only.to_csv(f'{dir_in}/processed/samples_not_in_UWM.csv', index=False)
df_inv_only.to_csv(f'{dir_in}/processed/samples_only_in_UWM.csv', index=False)

## 2. LMA samples should match with at least a OD or FF sample
n_LMA = df_inv.loc[df_inv['Type']=='LMA', 'Sample Number']
n_OD_FF = df_inv.loc[(df_inv['Type']=='OD') | ((df_inv['Type']=='FF')), 'Sample Number'].unique()
n_LMA_missing = set(n_LMA).difference(n_OD_FF)
# 394, missing FF; 719, flower, seed, npv; 137, missing FF or OD

## 3. cross comparing UWM sample and UCLA-Elsa samples
df_Elsa = pd.read_csv(f'{dir_in}/Lists_from_others/SHIFT_UCLA_JPL_Sample_IDs.csv')
n_Elsa = df_Elsa['JPL_sample_ID'].unique().astype(int)
n_inv_Elsa = set(n_inv).intersection(set(n_Elsa))
n_Elsa_only = set(n_Elsa).difference(n_inv_Elsa)
n_Elsa_LMA = set(n_inv_Elsa).difference(set(n_LMA.astype(int)))


## 4. Cross check sample file with the wetland samples
df_wetland = pd.read_csv(f'{dir_in}/Lists_from_others/SHIFT_wetland_samples_Silva.csv')
n_wetland = df_wetland['sample_id'].astype(int)
n_wetland_only = set(n_wetland).difference(set(n_sample))
n_wetland_only = set(n_wetland).difference(set(n_inv))

## 5. cross check samples in inv_only and elsa + wetland
n_inv_only_v2 = n_inv_only.difference(set(n_Elsa) | set(n_wetland))
# df_inv_only_update = df_inv[df_inv['Sample Number'].astype(int).isin(n_inv_only_v2)]
df_inv['Sample Number'] = df_inv['Sample Number'].astype(int)
df_inv_only_update = pd.DataFrame(data=n_inv_only_v2, columns=['Sample Number'])
df_inv_only_update.to_csv(f'{dir_in}/processed/samples_only_in_UWM_considered_Elsa_wetland.csv', index=False)

# figure out the sample type
df_inv_only_update_type = df_inv_only_update.merge(df_inv.loc[:, ['Sample Number', 'Type']], how='left', on='Sample Number')
len(df_inv_only_update_type.loc[df_inv_only_update_type['Type']!= 'BK', 'Sample Number'].unique())

## 6. cross check Anderegg samples with Dana's list
df_ander = pd.read_csv(f'{dir_in}/Lists_from_others/Anderegg_Trees_SHIFT_Samples.csv')
n_ander = df_ander.loc[df_ander['Sample #'].notna(), 'Sample #'].unique().astype(int)
# check if they are in Dana's list
n_ander_only = set(n_ander).difference(set(n_sample))
# check if they are in UWM list
n_ander_only_v2 = set(n_ander).difference(set(n_inv))

#%%------------------ for common species in both Dana's list and UWM's list, find the species distribution---------------
# get the common samples
df_sample_common = df_sample[df_sample['Sample Number'].isin(n_com)]
# drop NPV
df_sample_common = df_sample_common[df_sample_common['Species or type'] != 'NPV'].reset_index(drop=True)
# drop bulk
df_sample_common = df_sample_common[df_sample_common['Species or type'] != 'Bulk sample'].reset_index(drop=True)
# count sample numbers by species
df_sample_byspecies = df_sample_common.groupby('Species or type').count()
df_sample_byspecies = df_sample_byspecies.reset_index()

# clean the phenophase column
df_sample_common['Pheno'] = df_sample_common['Phenophase (if rare flowers or seeds, add as multi-select - if sampling flowers separately, add new entry)'].str.split(',').str.get(0)
df_sample_common_group = df_sample_common.groupby(['Species or type', 'Pheno']).agg({'Sample Number': ['count']})
df_sample_common_group = df_sample_common_group.reset_index()

df_sample_common_group.to_csv(f'{dir_in}/processed/common_samples_grouped.csv', index=False)