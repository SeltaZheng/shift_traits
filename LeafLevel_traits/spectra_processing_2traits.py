"""
1. Read .asd files into one
2. Calculate sample mean for the dry spec.
"""


#libraries:
import numpy as np
import pandas as pd
import os, glob
from specdal import Collection, Spectrum, read
import matplotlib.pyplot as plt
import seaborn as sns

def readdata(spec_fn, vectF, wl_left, wl_right, wl_step, spec_col):
    # Input: spec_fn: filename for the spec data; trait_fn: filename for the trait data
    # trait: a string for the name of the trait; vectF: True or False, do vector normalization or not
    # [wl_left: wl_right]: wavelength range
    # read trait X and spectra
    df_spec = pd.read_csv(spec_fn)
    # spectrum part of df_spec
    df_spec = df_spec.iloc[:, spec_col:]
    spec = df_spec.values

    wl = df_spec.columns.values.astype(float)

    ## If using NEON_v3 coefs:

    # wl = pd.read_csv(badbnd_fn).values

    wl = np.array(range(350, 2501, wl_step))

    # get rid of the bad bands before vector-normalization
    #    index = wl[:,1] == 0
    #    spec = spec[:,index]
    #    wl = wl[index,0]

    # normalize spectra
    if vectF:
        print(spec.shape)
        spec_len = np.tile(np.linalg.norm(spec, axis=1), (spec.shape[1], 1))
        spec = spec / spec_len.T

    # Extract spectra based on input intervals
    spec = spec[:, 0::wl_step]
    # extract spectra from a range, such as 1000 - 2400 nm
    index = np.logical_and(wl >= wl_left, wl <= wl_right)
    wl = wl[index]
    spec = spec[:, index]
    #####################################

    ## If using FFT coefs:

    #
    #    # get rid of the bad bands before vector-normalization
    ##    index = wl[:,1] == 0
    ##    spec = spec[:,index]
    ##    wl = wl[index,0]
    #
    #    # normalize spectra
    #    if vectF:
    #        print(spec.shape)
    #        spec_len = np.tile(np.linalg.norm(spec, axis=1), (spec.shape[1], 1))
    #        spec = spec/spec_len.T
    #        # Extract spectra based on input intervals
    #    # wl = pd.read_csv(badbnd_fn).values
    #    spec = spec[:,0::wl_step]
    #    wl = np.array(range(350,2501,wl_step))
    #
    #    # extract spectra from a range, such as 1000 - 2400 nm
    #    index = np.logical_and(wl>=wl_left, wl<=wl_right)
    #    wl = wl[index]
    #    spec = spec[:, index]
    #######################################

    print(spec.shape)
    return spec


#%%---------------- .asd reading and combining into one .csv--------------------------

Dir_spec = r'G:\Shared drives\Townsend-share\SampleProcessing\SHIFT\Spectra\FreezeDried'
Dir_out = r'G:\My Drive\Projects_ongoing\shift\data\spectra\FF'
# Dir_spec = r'G:\Shared drives\Townsend-share\SampleProcessing\SHIFT\Spectra\OvenDried'
# Dir_out = r'G:\My Drive\Projects_ongoing\shift\data\spectra\OD'




# Get all sub-folders:
subs = glob.glob(f'{Dir_spec}/*/')

# Loop through the sub folders:
for i, s in enumerate(sorted(subs)):

    # file name:
    fn = os.path.basename(os.path.normpath(s))

    # create a collection
    c = Collection(name=fn)

    for f in sorted(os.listdir(s)):
        spec = Spectrum(filepath=os.path.join(s, f))
        c.append(spec)

    # perform jump correction for the collection
    c.jump_correct(splices=[1000, 1800], reference=0)

    spec_data = c.data

    # plot the collection
    # c.plot(legend=False, ylim=(0, 1))

    # save the data to csv
    spec_data = spec_data.T
    # move index to a new column:
    spec_data.reset_index(inplace=True)
    spec_data = spec_data.rename(columns={'index': 'filename'})
    spec_data['handler'] = fn

    if i==0:
        df_spec = spec_data
    else:
        df_spec = pd.concat([df_spec, spec_data], ignore_index=True)

# df_spec.to_csv(f'{Dir_out}/ovendried_spectra.csv', index=False)
df_spec.to_csv(f'{Dir_out}/flashfrozen_spectra.csv', index=False)



#%%---------------- sample mean calculation-----------------------
Dir_in = r'G:\My Drive\Projects_ongoing\shift\data\spectra\OD'
df_spec = pd.read_csv(f'{Dir_in}/ovendried_spectra.csv')

# filter out bad spectra:
idx = df_spec[df_spec['350.0'] > 0.9].index
df_spec.drop(idx, inplace=True)



# ID column:
df_spec['sample_ID'] = df_spec['filename'].str[0:-6]
df_spec = df_spec.drop(columns=['filename'])
spec_sample = df_spec.groupby(['sample_ID']).mean()
spec_sample.reset_index(inplace=True)
spec_sample = spec_sample.rename(columns={'index': 'sample_ID'})
spec_sample.to_csv(f'{Dir_in}/ovendried_spectra_sample_mean.csv', index=False)

# Plot the mean spec
for i in range(403):
    plt.plot(range(len(spec_sample.columns)-1), spec_sample.iloc[i, 1:])
plt.savefig(f'{Dir_in}/spec_sample_mean.png', dpi=300)

#%%---------------- trait prediction using NEON leaf level model------------------------------
Dir_Coef_All = r'G:\My Drive\Data_Organization\LeafLevel\PLSR\Dry'
Dir_spec = r'G:\My Drive\Projects_ongoing\shift\data\spectra\OD'
Dir_out = r'G:\My Drive\Projects_ongoing\shift\data\traits'

# define a new function to locate the right columns:
# find = lambda c_name, wl: [[i for i, x in enumerate(c_name) if x == e] for e in wl]

# set up directories for models and leaf spectras
# Dir_Coef_All = 'C:/Users/tzheng39/Documents/Working/Projects_ongoing/IndianProject/Sample_dryspec/DrySpecCoef/FFT/'
# Dir_Coef_All = 'C:/Users/tzheng39/Documents/Working/Projects_ongoing/IndianProject/Sample_dryspec/DrySpecCoef/NEON_LT/'

# set up the base name for trait coefs
# coef_base = 'PLSR_500_raw_coef_'
coef_base = 'PLSR_coef_raw_'
pred_base_all = 'shift_DS_Predict_NEON_v3.csv'
Spec_fn = 'ovendried_spectra_sample_mean.csv'

spec_col = 1

# set up the traits to predict
trait = ['0_Cellulose', '0_Lignin', '0_Fiber', '0_Starch', '0_Sugar', '0_NSC', '0_Phenolics', '0_Flavonoids',
         '0_Carbon', '0_d13C', '0_d15N', '0_Nitrogen',
         '0_Phosphorus', '0_Potassium', '0_Calcium', '0_Magnesium', '0_Sulfur', '0_Aluminum', '0_Boron', '0_Copper',
         '0_Iron', '0_Manganese', '0_Zinc']

trait_new = [x.split('_')[1] for x in trait]

# set up the wavelength for the PLSR coefs
wl_l = 400
wl_r = 2400
wl_step = 5
vectF = True

# Dataframe for the prediction

i = 0

#  Read the coef for the trait and predict new trait based on Spec
for t in trait:
    spec_fn = f'{Dir_spec}/{Spec_fn}'
    spec = readdata(spec_fn, vectF, wl_l, wl_r, wl_step, spec_col)
    # add one extra column in front of Spec to consider the intercept of the PLS model
    x = np.zeros((len(spec), 1)) + 1
    spec = np.c_[x, spec]

    # using the coefs derived using all plots
    coef_fn_all = f'{coef_base}{t}_{wl_l}_{wl_r}.csv'
    # coef_fn_all = coef_base + t+'.csv'
    t_coef = pd.read_csv(f'{Dir_Coef_All}/{coef_fn_all}')
    t_coef = t_coef.values
    t_pred = np.matmul(spec, t_coef)
    t_mean = np.mean(t_pred, axis=1)
    t_std = np.std(t_pred, axis=1)
    t_new = trait_new[i]
    if i == 0:
        DS_traits = np.zeros((len(spec), len(trait) * 2))
        header = []

    DS_traits[:, (i * 2)] = t_mean
    DS_traits[:, (i * 2 + 1)] = t_std
    #    ratio = DS_traits[:,i+1]/abs(DS_traits[:,i])
    #    DS_traits[ratio>0.25,i] = np.nan
    if t_new == 'd13C':
        DS_traits[DS_traits[:, (i * 2)] > 0, i * 2] = np.nan
    elif (t_new != 'd13C') & (t_new != 'd15N'):
        DS_traits[DS_traits[:, (i * 2)] < 0, i * 2] = np.nan
    header.append(t_new + '_M')
    header.append(t_new + '_SD')
    i = i + 1

DS_traits = pd.DataFrame(DS_traits, columns=header)
DS_traitmeta = pd.read_csv(f'{Dir_spec}/{Spec_fn}')
DS_traits = pd.concat([DS_traitmeta.iloc[:, 0:spec_col], DS_traits], axis=1)
DS_traits.to_csv(f'{Dir_out}/{pred_base_all}', index=False)



#%%---------------- plot the predicted traits -------------------------------
trait_ls = sorted(trait_new)
fig, axs = plt.subplots(figsize=(15, 15), ncols=5, nrows=5)
axes = axs.ravel()
for i, t in enumerate(trait_ls):
    sns.histplot(DS_traits, x=f'{t}_M', ax=axes[i])
# remove extra axes:
for ax in axs[4, 3:]:
    ax.remove()

fig.tight_layout()
fig.savefig(f'{Dir_out}/DS_predicted_mean.png', dpi=200)

# second figure for uncertainty:
plt.close(fig)
# calculate the uncertainty:
col_mean = [f'{x}_M' for x in trait_ls]
col_sd = [f'{x}_SD' for x in trait_ls]
uncertainty = abs(DS_traits.loc[:, col_sd].values/DS_traits.loc[:, col_mean].values)*100
df_unc = pd.DataFrame(data=uncertainty, columns=trait_ls)

# calculate average of uncertainty for 95% data:
pct_cut = np.nanpercentile(uncertainty, 95, axis=0)
uncertainty_cut = np.copy(uncertainty)
uncertainty_cut[uncertainty >= pct_cut] = np.nan
# avg_unc = np.nanmean(uncertainty_cut, axis=0)
df_unc_cut = pd.DataFrame(data=uncertainty_cut, columns=trait_ls)

fig, axs = plt.subplots(figsize=(15, 15), ncols=5, nrows=5)
axes = axs.ravel()
for i, t in enumerate(trait_ls):
    sns.histplot(df_unc_cut, x=t, ax=axes[i])

# first remove the original axes:
gs = axs[4, 3].get_gridspec()
for ax in axs[4, 3:]:
    ax.remove()
axbig = fig.add_subplot(gs[4, 3:])
# box plot for all

# melt the dataframe:
df_melt = pd.melt(df_unc_cut, var_name='Trait', value_name='Uncertainty')
sns.boxplot(data=df_melt, x='Trait', y='Uncertainty', ax=axbig)
axbig.set_xticklabels(axbig.get_xticklabels(), rotation=90)
fig.tight_layout()
fig.savefig(f'{Dir_out}/DS_predicted_sd.png', dpi=200)





