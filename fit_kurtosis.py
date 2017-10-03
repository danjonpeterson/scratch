#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='fit a kurtosis tensor from diffusion data')
# required args
parser.add_argument('-d','--data',required=True, help='4D diffusion image')
parser.add_argument('-b','--bvals',required=True, help='b-values text file')
parser.add_argument('-v','--bvecs',required=True, help='b-vectors text file')

# optional args
parser.add_argument('-t','--testing',action='store_true', help='set this flag when testing - does fast but incomplete processing (output file names and types are correct)')
parser.add_argument('-o','--outdir',help='output directory',default='.')

args=parser.parse_args()
# Line for testing in ipython notebook
# args=parser.parse_args('-d','out/smoothed_raw_diffusion.nii.gz','-b','bvals.txt','-v','bvecs.txt','-t')

print('Reading Gradient Table')
from dipy.data import gradient_table
gtab = gradient_table(args.bvals,args.bvecs)

print('Loading Diffusion Data')
import nibabel as nib
img = nib.load(args.data)

data = img.get_data()
affine = img.get_affine()

print('Generating Mask w/ Median Otsu method')
from dipy.segment.mask import median_otsu
maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)

if args.testing:
    axial_slice = 16


print('Denoising Data')
from dipy.denoise.noise_estimate import estimate_sigma
sigma = estimate_sigma(data, N=4)

import numpy as np
mask_roi = np.zeros(data.shape[:-1], dtype=bool)

if args.testing:
    mask_roi[:, :, axial_slice] = mask[:, :, axial_slice]
else:
    mask_roi[:, :, :] = mask[:, :, :]

from dipy.denoise.nlmeans import nlmeans
den = nlmeans(data, sigma=sigma, mask=mask_roi)

if args.testing:
    den = den[:, :, axial_slice, :]
else:
    den = den[:, :, :, :]

print('Fitting the Kurtosis Model')
import dipy.reconst.dki as dki
dkimodel = dki.DiffusionKurtosisModel(gtab)

dkifit = dkimodel.fit(den)


print('Calculating the Contrasts')
MK = dkifit.mk(min_kurtosis=-1e9,max_kurtosis=1e9)
AK = dkifit.ak(min_kurtosis=-1e9,max_kurtosis=1e9)
RK = dkifit.rk(min_kurtosis=-1e9,max_kurtosis=1e9)

print('Saving the Output')

f=open(args.outdir+'/sigma.txt','w')
f.write('Estimated noise levels by dipy.denoise.noise_estimate: \n')
for n in sigma:
    f.write(str(n)+'\n')
f.close()

MKimg=nib.Nifti1Image(MK,affine)
nib.save(MKimg,args.outdir+'/MK.nii.gz')

AKimg=nib.Nifti1Image(AK,affine)
nib.save(AKimg,args.outdir+'/AK.nii.gz')

RKimg=nib.Nifti1Image(RK,affine)
nib.save(RKimg,args.outdir+'/RK.nii.gz')

Wimg=nib.Nifti1Image(dkifit.kt,affine)
nib.save(Wimg,args.outdir+'/kurtosis_tensor.nii.gz')




#import matplotlib.pyplot as plt

#fig2, ax = plt.subplots(1, 3, figsize=(12, 6),
#                        subplot_kw={'xticks': [], 'yticks': []})

#fig2.subplots_adjust(hspace=0.3, wspace=0.05)

#ax.flat[0].imshow(MK, cmap='gray')
#ax.flat[0].set_title('MK')
#ax.flat[1].imshow(AK, cmap='gray')
#ax.flat[1].set_title('AK')
#ax.flat[2].imshow(RK, cmap='gray')
#ax.flat[2].set_title('RK')

#plt.show()
