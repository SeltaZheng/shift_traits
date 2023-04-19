'''
Sample selection using Kennard-Stone algorithm and SPXY algorithm
code adapted from https://hxhc.xyz/post/kennardstone-spxy/#spxy-split
Algorithm based on
Galvao, Roberto Kawakami Harrop, et al. "A method for calibration and validation subset partitioning." Talanta 67.4 (2005): 736-740.
Li, Wenze, et al. "HSPXY: A hybrid‐correlation and diversity‐distances based data partition method." Journal of Chemometrics 33.4 (2019): e3109
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


def random_split(spectra, test_size=0.25, random_state=None, shuffle=True, stratify=None):
    """implement random_split by using sklearn.model_selection.train_test_split function. See
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    for more infomation.
    """
    return train_test_split(
        spectra,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify)


def kennardstone(spectra, test_size=0.25, metric='euclidean', *args, **kwargs):
    """Kennard Stone Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    --------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """

    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        distance = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts


def spxy(spectra, yvalues, test_size=0.25, metric='euclidean', *args, **kwargs):
    """SPXY Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    ---------
    Galvao et al. (2005). A method for calibration and validation subset partitioning.
    Talanta, 67(4), 736-740. (https://www.sciencedirect.com/science/article/pii/S003991400500192X)
    """

    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        yvalues = yvalues.reshape(yvalues.shape[0], -1)
        distance_spectra = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        distance_y = cdist(yvalues, yvalues, metric=metric, *args, **kwargs)
        distance_spectra = distance_spectra / distance_spectra.max()
        distance_y = distance_y / distance_y.max()

        distance = distance_spectra + distance_y
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts


def max_min_distance_split(distance, train_size):
    """sample set split method based on maximun minimun distance, which is the core of Kennard Stone
    method
    Parameters
    ----------
    distance : distance matrix
        semi-positive real symmetric matrix of a certain distance metric
    train_size : train data sample size
        should be greater than 2
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero-based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero-based
    """

    select_pts = []
    remaining_pts = [x for x in range(distance.shape[0])]

    # first select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    select_pts.append(first_2pts[0])
    select_pts.append(first_2pts[1])

    # remove the first 2 points from the remaining list
    remaining_pts.remove(first_2pts[0])
    remaining_pts.remove(first_2pts[1])

    for i in range(train_size - 2):
        # find the maximum minimum distance
        select_distance = distance[select_pts, :]
        min_distance = select_distance[:, remaining_pts]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts


## main:
dir_spec = r'D:\GoogleDrive\Projects_ongoing\shift\data\spectra\OD'
dir_t = r'D:\GoogleDrive\Projects_ongoing\shift\data\traits'
dir_out =r'D:\GoogleDrive\Projects_ongoing\shift\data\meta'

df_t = pd.read_csv(f'{dir_t}/shift_DS_Predict_NEON_v3.csv')
df_spec = pd.read_csv(f'{dir_spec}/ovendried_spectra_sample_mean.csv')

## parse the sample ID, exclude NPVs
dir_in = r'D:\GoogleDrive\Projects_ongoing\shift\data\meta'
# df_sample = pd.read_csv(f'{dir_in}/processed/sample_list_simplified.csv')
df_iform = pd.read_csv(f'{dir_in}/processed/sample_list_simplified_ORNL_v3.csv', encoding='unicode_escape')
df_inv = pd.read_csv(f'{dir_in}/processed/SHIFT_sample_inventory_20230308.csv')
# Clean the df_inv
df_inv = df_inv[df_inv['Sample Number'] != '131_flower']
df_inv = df_inv[df_inv['Sample Number'] != '161_S'].reset_index(drop=True)
n_inv = df_inv['Sample Number'].unique().astype(int)
n_inv_OD = df_inv.loc[(df_inv['Type']=='OD') | (df_inv['Type']=='BK'), 'Sample Number'].unique().astype(int)
# match the spec sample with inv
sample_n = df_spec['sample_ID'].str.split('_').str.get(1).astype(int)
diff_spec2inv = set(sample_n).difference(n_inv) # {1824, 1678, 531, 1689, 1693, 1694}
diff_inv2spec = set(n_inv_OD).difference(sample_n) # { 551, 40, 1601, 71, 174, 1776, 755,  1859, 335, 336, 346,  374, 376,  449, 451, 1012}

# filter out NPV based on df_iform
spec_meta = pd.DataFrame(df_spec['sample_ID'])
spec_meta['Sample Number'] = sample_n
NPV = df_iform.loc[(df_iform['Species or type'] == 'NPV') & (df_iform['Sample Taken?'] == 'Yes'), 'Sample Number']
# filter out flower/seeds/full senescence samples
# clean the phenophase column
df_iform['Pheno'] = df_iform['Phenophase (if rare flowers or seeds, add as multi-select - if sampling flowers separately, add new entry)'].str.split(',').str.get(0)
dropV = df_iform.loc[(df_iform['Pheno'] == 'Flowers') | (df_iform['Pheno'] == 'Full senescence') | (df_iform['Pheno'] == 'Seeds'), 'Sample Number']
# drop nan and convert to int
NPV = list(NPV[~NPV.isna()].astype(int))
# find bulk from inventory
BK = df_inv.loc[df_inv['Type']=='BK', 'Sample Number'].unique().astype(int)
# no scan list from Natalie
df_noScan = pd.read_csv(f'{dir_out}/Lists_from_others/sample_no_LMA_scans.csv')
noScan = df_noScan['no scans'].astype(int)

drop_ls = NPV + list(dropV[~dropV.isna()].astype(int)) + list(BK) + list(noScan)
idx = ~spec_meta['Sample Number'].isin(drop_ls)
spec_meta_veg = spec_meta[idx].reset_index(drop=True)
spec_veg = df_spec[idx].iloc[:, 1:].values


# select and match the corresponding traits
trait_ls = ['Cellulose',  'Fiber', 'Lignin', 'Nitrogen', 'Calcium', 'NSC',  'Phenolics']
trait_m = [x + '_M' for x in trait_ls]

traits = df_t.loc[idx, trait_m].values

# select 200 samples
ratio = 1 - 200/spec_veg.shape[0]

x_train_index, x_test_index = kennardstone(spec_veg, test_size=ratio)

spec_ls = spec_meta_veg.iloc[x_train_index, :]


# combine spec and traits
# remove the traits record with nan
idx_t = ~np.isnan(traits).any(axis=1)
ratio = 1 - 200/spec_veg[idx_t, :].shape[0]
x_train_index, x_test_index = spxy(spec_veg[idx_t, :], traits[idx_t, :], test_size=ratio)

spec_meta_veg_sub = spec_meta_veg[idx_t].reset_index(drop=True)
spec_t_ls = spec_meta_veg_sub.iloc[x_train_index, :]
common_n = set(spec_ls['Sample Number']).intersection(set(spec_t_ls['Sample Number']))
common_ls = spec_ls[spec_ls['Sample Number'].isin(common_n)]
# perform join to get more info

spec_ls_meta = spec_ls.merge(df_iform, how='left', on='Sample Number')
spec_t_ls_meta = spec_t_ls.merge(df_iform, how='left', on='Sample Number')
common_ls_meta = common_ls.merge(df_iform, how='left', on='Sample Number')

# add more info from Elsa and wetland
df_elsa = pd.read_csv(f'{dir_out}/Lists_from_others/SHIFT_UCLA_JPL_Sample_IDs.csv')
df_wetland = pd.read_csv(f'{dir_out}/Lists_from_others/SHIFT_wetland_samples_Silva.csv')
df_wetland.rename(columns={'sample_id': 'Sample Number'}, inplace=True)
df_elsa.rename(columns={'JPL_sample_ID': 'Sample Number'}, inplace=True)

spec_ls_meta = spec_ls_meta.merge(df_elsa.loc[:, ['Sample Number', 'species_code']].drop_duplicates(), how='left', on='Sample Number')
spec_ls_meta = spec_ls_meta.merge(df_wetland.loc[:, ['Sample Number', 'species']].drop_duplicates(), how='left', on='Sample Number')

spec_t_ls_meta = spec_t_ls_meta.merge(df_elsa.loc[:, ['Sample Number', 'species_code']].drop_duplicates(), how='left', on='Sample Number')
spec_t_ls_meta = spec_t_ls_meta.merge(df_wetland.loc[:, ['Sample Number', 'species']].drop_duplicates(), how='left', on='Sample Number')

common_ls_meta = common_ls_meta.merge(df_elsa.loc[:, ['Sample Number', 'species_code']].drop_duplicates(), how='left', on='Sample Number')
common_ls_meta = common_ls_meta.merge(df_wetland.loc[:, ['Sample Number', 'species']].drop_duplicates(), how='left', on='Sample Number')

spec_ls_meta.to_csv(f'{dir_out}/selected_sample_list_based_on_dryspec_v2.csv', index=False)
spec_t_ls_meta.to_csv(f'{dir_out}/selected_sample_list_based_on_dryspec_traits_v2.csv', index=False)
common_ls_meta.to_csv(f'{dir_out}/selected_sample_list_in_common_v2.csv', index=False)

#%%----------------- With the final list, check if the sample has both OD and FF parts-------------------------------
dir_in =r'D:\GoogleDrive\Projects_ongoing\shift\data\meta'
df_od_final = pd.read_csv(f'{dir_out}/selected_samples_final_correctdate.csv')
df_inv = pd.read_csv(f'{dir_in}/processed/SHIFT_sample_inventory_20230308.csv')
# Clean the df_inv
df_inv = df_inv[df_inv['Sample Number'] != '131_flower']
df_inv = df_inv[df_inv['Sample Number'] != '161_S'].reset_index(drop=True)

# FF sample list from df_inv
n_ff = df_inv.loc[df_inv['Type']=='FF', 'Sample Number'].astype(int)
n_od_final = df_od_final['sample number'].astype(int)
n_common = set(n_od_final).intersection(n_ff)
n_dif = set(n_od_final).difference(n_ff)
df_common = df_od_final[df_od_final['sample number'].isin(n_common)]
df_common.to_csv(f'{dir_out}/selected_sample_final_wt_od_ff.csv', index=False)
pd.DataFrame(n_dif).to_csv(f'{dir_out}/no_FF_samples.csv')

# Overall comparison
df_iform = pd.read_csv(f'{dir_in}/processed/sample_list_simplified_ORNL_v3.csv', encoding='unicode_escape')
NPV_BK = df_iform.loc[((df_iform['Species or type'] == 'NPV') | (df_iform['Species or type'] == 'Bulk sample')) & (df_iform['Sample Taken?'] == 'Yes'), 'Sample Number']
NPV_BK = list(NPV_BK[~NPV_BK.isna()].astype(int))

# only keep od and FF from inventory
df_inv = df_inv[(df_inv['Type']=='OD') | (df_inv['Type']=='FF')]
# drop NPV from inventory
idx = ~df_inv['Sample Number'].astype(int).isin(NPV_BK)
df_inv = df_inv[idx]

n_od = df_inv.loc[df_inv['Type']=='OD', 'Sample Number'].unique().astype(int)
n_ff = df_inv.loc[df_inv['Type']=='FF', 'Sample Number'].unique().astype(int)

n_com = set(n_ff).intersection(set(n_od))
n_ff_only = set(n_ff).difference(set(n_od))
n_od_only = set(n_od).difference(set(n_ff))

pd.DataFrame(n_ff_only).to_csv(f'{dir_in}/UWM_inv_sort/FF_only_in_inv_exclude_NPV_BK.csv', index=False)
pd.DataFrame(n_od_only).to_csv(f'{dir_in}/UWM_inv_sort/OD_only_in_inv_exclude_NPV_BK.csv', index=False)
pd.DataFrame(n_com).to_csv(f'{dir_in}/UWM_inv_sort/OD_and_FF_in_inv_exclude_NPV_BK.csv', index=False)

#%%-------------------------Check the final list against the most recent flash frozen list----------------
dir_in =r'D:\GoogleDrive\Projects_ongoing\shift\data\meta'
df_od_final = pd.read_csv(f'{dir_in}/selected_samples_final_correctdate.csv')
df_od_done = pd.read_csv(f'{dir_in}/selected_sample_final_wt_od_ff.csv')
df_ff = pd.read_csv(f'{dir_in}/raw/SHIFT_flash_frozen_UWM.csv',  encoding='unicode_escape')
n_final = df_od_final['sample number'].astype(int)
n_ff = df_ff['sample number'].astype(int)
n_done = df_od_done['sample number'].astype(int)
n_com = set(n_final).intersection(set(n_ff))
n_dif = set(n_final).difference(set(n_ff)) # {384, 1665, 259, 37, 933, 1531, 1208, 1529, 1659, 125, 1662}
n_todo = set(n_com).difference(set(n_done))
df = pd.DataFrame(data=n_todo, columns=['sample number'])
df.to_csv(f'{dir_in}/samples_todo.csv', index=False)

#%%---------------------------- select N/15N/13C samples within the sample_todo + wt_od_ff list
dir_in =r'D:\GoogleDrive\Projects_ongoing\shift\data\meta'
dir_t = r'D:\GoogleDrive\Projects_ongoing\shift\data\traits'
dir_spec = r'D:\GoogleDrive\Projects_ongoing\shift\data\spectra\OD'

df_od_done = pd.read_csv(f'{dir_in}/selected_sample_final_wt_od_ff.csv')
df_ff = pd.read_csv(f'{dir_in}/samples_todo.csv')
df_t = pd.read_csv(f'{dir_t}/shift_DS_Predict_NEON_v3.csv')
df_spec = pd.read_csv(f'{dir_spec}/ovendried_spectra_sample_mean.csv')
df_spec['sample number'] = df_spec['sample_ID'].str.split('_').str.get(1).astype(int)

# target list
tgt = list(df_od_done['sample number'].astype(int)) + list(df_ff['sample number'].astype(int))
idx = df_spec['sample number'].isin(tgt)
df_trait = df_t.loc[idx, ['d13C_M', 'd15N_M', 'Nitrogen_M']].reset_index(drop=True)
tgt_sample = df_spec.loc[idx, 'sample number'].reset_index(drop=True)
traits = df_trait.values
ratio = 1 - 36/traits.shape[0]

x_train_index, x_test_index = kennardstone(traits, test_size=ratio)
select_traits = traits[x_train_index, :]
sample_select = tgt_sample[x_train_index]
sample_select = pd.DataFrame(data=sample_select, columns=['sample number'])
sample_select.to_csv(f'{dir_in}/36_isotope_samples.csv', index=False)
