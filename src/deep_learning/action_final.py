#%%
import numpy as np
import pandas as pd
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import os
import sys


#%%
def calculate_max_min(df):
    max_val=df.to_numpy().max()
    min_val=df.to_numpy().min()
    return max_val, min_val

#%%
def min_max_scaling(df,max_val,min_val):
    df=(df-min_val)/(max_val-min_val)
    return df


# %%
# csv_file_name='shortened_IMG_0555_4_best_hybrid_model_vgg19_prediction_parallel.csv'
csv_file_name='IMG_0882_60_best_hybrid_model_vgg19_prediction_parallel.csv'
csv_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(csv_files_path,csv_file_name)
df = pd.read_csv(csv_file,index_col=0) 
# print(df)
# %%
vec0=df[df.columns[3]]
vec1=df[df.columns[6]]
# vec2=df[df.columns[2]]
# vec3=df[df.columns[3]]
# %%
low_y=df[df.columns[8]].argmin()
high_y=df[df.columns[8]].argmax()
# d=vec1.iloc[low_y:,]
d=vec1.iloc[low_y:high_y+1,]
d = d[d!= -1]
# dsub=d-d.shift(1)
dsub=abs(d.pct_change())
dsub=dsub.dropna(axis=0)
# max_val, min_val=calculate_max_min(dsub)
# dsub=min_max_scaling(d,max_val,min_val)
sigma=1.5
# res_0 = abs(ndimage.gaussian_laplace(vec0, sigma=sigma))
res_1 = abs(ndimage.gaussian_laplace(dsub, sigma=sigma))
res_1=pd.Series(res_1,index=dsub.index)
# calculate interquartile range
# q25, q75 = np.percentile(res_1, 25), np.percentile(res_1, 75)
# iqr = q75 - q25
# # calculate the outlier cutoff
# cut_off = iqr * 1.5
# lower, upper = q25 - cut_off, q75 + cut_off
# # identify outliers
# outliers = [x for x in res_1 if x < lower or x > upper]
# ball_release=res_1[res_1==outliers[0]].index.tolist()
# print(ball_release)


q25, q75 = np.percentile(dsub, 25), np.percentile(dsub, 75)
iqr = q75 - q25
# cut_off = iqr *np.median(dsub)*13.5
cut_off = iqr * 20
lower, upper = q25 - cut_off, q75 + cut_off
outliers = [x for x in dsub if x < lower or x > upper]
outliers=np.array(outliers)
outliers=outliers[outliers!=0]
d=vec1.iloc[low_y-2:,]
if outliers.shape[0]!=0:
    outlier_in_wrist=dsub[dsub==outliers[0]].index.tolist()
    print(outlier_in_wrist)
    d=d.drop(index=outlier_in_wrist)

#%%
# d=vec1.iloc[low_y-2:,]
# # d=vec1
# d=d.drop(index=outlier_in_wrist)
# d = d[d!= -1]
# # dsub=abs(d.pct_change())
# dsub=d-d.shift(1)
# dsub=dsub.dropna(axis=0)
# max_val, min_val=calculate_max_min(dsub)
# dsub=min_max_scaling(d,max_val,min_val)
# q25, q75 = np.percentile(dsub, 25), np.percentile(dsub, 75)
# iqr = q75 - q25
# cut_off = iqr *np.median(dsub)*13.5
# # cut_off = iqr * 20
# lower, upper = q25 - cut_off, q75 + cut_off
# outliers = [x for x in dsub if x < lower or x > upper]
# outliers=np.array(outliers)
# outliers=outliers[outliers!=0]
# if outliers.shape[0]!=0:
#     ball_release=dsub[dsub==outliers[0]].index.tolist()
#     print(ball_release)

#%%

d = d[d!= -1]
dsub=abs(d-d.shift(1))
# dsub=abs(d.pct_change())
dsub=dsub.dropna(axis=0)
max_val, min_val=calculate_max_min(dsub)
dsub=min_max_scaling(dsub,max_val,min_val)
res_1 = abs(ndimage.gaussian_laplace(dsub, sigma=sigma))
res_1=pd.Series(res_1,index=dsub.index)
q25, q75 = np.percentile(res_1, 25), np.percentile(res_1, 75)
iqr = q75 - q25
# calculate the outlier cutoff
cut_off = iqr * np.median(res_1)*13.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in res_1 if x > upper]
outliers=np.array(outliers)
outliers=outliers[outliers!=0]
if outliers.shape[0]!=0:
    ball_release=res_1[res_1==outliers[0]].index.tolist()
    print(ball_release)

#%%
vec2=df[df.columns[7]]
vec2=vec2.dropna()
d=vec2.iloc[low_y:high_y+1,]
foot_plant=d.idxmin()
print(foot_plant)


#%%
sp=df[df.columns[0]]
sp=sp.iloc[:low_y]
sp=sp[sp>0.6]
if sp[sp>0.9].shape[0]!=0:
    set_position=sp[sp>0.9].index[0]
else:
    set_position=sp.idxmax()
print(set_position)
forward_motion=sp[sp<0.6].index[0]
print(forward_motion)
#%%


