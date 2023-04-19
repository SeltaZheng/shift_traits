# -*- coding: utf-8 -*-
"""
Based on nliu58@wisc.edu's code
calculate leaf area based on leaf scans
"""

import numpy as np
import glob,os
import pandas as pd
from PIL import Image
import matplotlib.colors as color
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def process1(imgFN):
    #imgFN=r'D:\conifer_needle_scans\20170605\LALA_SU_1_201706050001.tif'
    #read rgb
    print(imgFN)
    img=Image.open(imgFN)
    dpi=img.info['dpi'][0]
    r,g,b= img.split()
    img.close()
    
    #rgb->hsv
    r=np.array(r.convert('L'))/255.0
    g=np.array(g.convert('L'))/255.0
    b=np.array(b.convert('L'))/255.0
    rgb=np.dstack((r,g,b))
    hsv=color.rgb_to_hsv(rgb)
    h=hsv[:,:,0]*360
    #s=hsv[:,:,1]*100
    #v=hsv[:,:,2]*100
    
    # Plot histgram of the image
    # plt.figure()
    # n, bins, patches =plt.hist(h.flatten(),10)
    
#    plt.figure()
#    n, bins, patches =plt.hist(r.flatten(),10)
#    plt.figure()
#    n, bins, patches =plt.hist(h.flatten(),10)
#    plt.figure()
#    n, bins, patches =plt.hist(b.flatten(),10)
#    plt.show()
    
    #extract green pixels based on hue values
#    mask=np.logical_and(h>20,h<130)  # over estimate leaf area
    mask=np.logical_and(h>35,h<160) #  correct overestimated leaf area a bit
    #mask=np.logical_and(h>17,h<160) # For redish leaves such as young red maple 
    # mask=np.logical_and(h>0,h<160) # For tamarack needles
    # mask=np.logical_and(h>17,h<160) # Correct for redish colored leaves, also for leaves with big holes,for most R3 samples
    # mask=np.logical_and(h>10,h<160) # 329_1619_T 02,03,04 , Fill holes R3.
    # mask=np.logical_and(r<0.419,b<0.451) # For 283_1526,242_1U_B 01,02 purple ash; R2 271_T1 with brown spots
    # mask=np.logical_and(h>17,h<160) # also for R3 needles.
#    mask=np.logical_and(h>50,h<200) # for whitish conifer needles: blue spruce with fill wholes
#    mask=np.logical_and(h>27,h<235) # for whitish conifer needles: firs without fill wholes
#    mask=np.logical_and(h>10,h<265) # for super white needles such as D00_0013_C1, 
#    mask=np.logical_and(h>10,h<275)  # for some conifers in D19
#    print np.min(r),np.max(r)
#    mask=np.logical_and(r<0.5,b<1)  # for purple leaves such as D00_0010
#    mask=np.logical_and(h>10,h<200)  # for hancock dark agriculture leaves
#    mask=np.logical_and(h>10,h<200)  # for California bad ones 
#    mask=np.logical_and(h>15,h<160)   # for D18, yellowish moss
#    mask=np.logical_and(h>10,h<160)  # for D18, dark leaves
#    mask=np.logical_and(h>15,h<200)   # for D19, yellowish moss
    # mask=np.logical_and(h>15,h<180)   # for D19,  purple-corrected images(green corrections)
#    mask=np.logical_and(r<0.8,b>0.01176) # for D19,purple
#    mask=np.logical_and(h>=0,h<100) # for D19,purple
#    mask1=np.logical_and(r<0.8,b>0.01176) # for D19,purple
    # mask2=np.logical_and(h>20,h<130)
#    mask = np.logical_or(mask1,mask2)
#    mask=~mask

    
    #remove single noisy pixels
    strc=np.array([[0,0,0],[0,1,0],[0,0,0]])
    c=ndimage.binary_hit_or_miss(mask, structure1=strc)
    c=(~c).astype(int)
    mask=mask*c
    
    #remove holes
    # mask=ndimage.binary_opening(mask, structure=np.ones((2,2))).astype(np.int)
    # mask=ndimage.binary_fill_holes(mask).astype(np.int) 
    
    area=np.sum(mask)
    
    area=(np.sum(mask)*1.0/dpi/dpi)*2.54*2.54#1 square inch = 2.54*2.54 square centimeters
    
    mask=Image.fromarray(np.uint8(mask)*255)    
    maskFN=imgFN[:-5]+'_mask.jpg' # name the mask
    mask.save(maskFN)
    return area
    
def process2(inDir):
    imgFNs=glob.glob(inDir+'/*.jpeg')
    imgFNs.extend(glob.glob('*.tif'))
    imgFNs.extend(glob.glob('*.bmp'))
    
    
    if imgFNs==[]:
        pass
    
    area=[]
    sample_id=list()
    for imgFN in imgFNs:
        print(imgFN)
        area.append(process1(imgFN))
        sample_id.append(os.path.basename(imgFN)[:-4])
    
    df=pd.DataFrame()
    df['filename']=imgFNs
    df['Area(cm^2)']=area
    df['sample_id']=sample_id
    return df

def main(inDir,outDir):
    # theDirs=glob.glob(os.path.join(inDir,'*'))
    theDirs = inDir
    outDF=pd.DataFrame()
    # for theDir in theDirs:
    #     df=process2(theDir)
    #     outDF=pd.concat((outDF,df))
    outDF = process2(inDir)
    fname, ext = os.path.splitext(theDirs)  # by ZWang
    outFN=os.path.join(outDir,fname.split('\\')[-1]+'_results.csv')  # by ZWang
#    outFN=os.path.join(outDir,'results.csv')
    outDF.to_csv(outFN,index=False)

inDir = r'D:\GoogleDrive\Projects_ongoing\shift\data\LMA\LMA_scans_renamed'

# Make the output directory:
# os.mkdir(inDir + '/Area') 
# outDir = inDir + '/Area'
outDir = r'D:\GoogleDrive\Projects_ongoing\shift\data\LMA'
    
main(inDir,outDir)


#%%--------------include sample number column------------------
df_in = pd.read_csv(f'{outDir}/LMA_scans_renamed_results.csv')
df_in['sample number'] = df_in['sample_id'].str.split('_').str.get(0).astype(int)
df_in.to_csv(f'{outDir}/LMA_scans_renamed_area.csv')