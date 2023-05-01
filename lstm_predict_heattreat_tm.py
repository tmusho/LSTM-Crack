# -*- coding: utf-8 -*-
"""LSTM_Predict_HeatTreat_TM.ipynb

Author: Dr. Terence Musho - West Virginia University
        Jacob Keesler-Evans - West Virginia University
        
Description: This code predicts new data based on trained model. You can provide intermediate
             stress intensity and times. This code uses the weights trained in the lstm_train_heattreat_tm.py code.

Original file is located at
    https://colab.research.google.com/drive/1_2ZwyC7diHwgu_KxMGwBpVXgxhL4OrpJ
"""

# Commented out IPython magic to ensure Python compatibility.
# %reset #clear the ipython workspace variables

# Commented out IPython magic to ensure Python compatibility.
#select that same tensorflow version as Training Script
# %tensorflow_version 2.x

#Import Statements
import os
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
from scipy.optimize import curve_fit

from numpy import array
import joblib
import sys

import math
from math import sqrt

import tensorflow as tf
from tensorflow.python.client import device_lib

#let ping the GPU and see what colab gave us...
print(tf.test.gpu_device_name())
print(device_lib.list_local_devices())

#let's also check version of tf and if eager is on.
print(tf.executing_eagerly())
print(tf.__version__)
print(keras.__version__)

#Change plot settings
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 15
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2

#data is on google drive so need to load
from google.colab import drive
drive.mount('/content/drive')

#this line needs to be changed depending on file structure of google drive
#os.chdir('/content/drive/MyDrive/WVU_GradStudents_ShareFolder/Jacob_Research/New LSTM Crack')
os.chdir('/content/drive/MyDrive/New LSTM Crack/')

"""# Functions"""

def read_file(file_name):
  file = file_name
  return pd.read_csv(file)

def create_prediction_data(prediction_data):
  data_array = np.asarray(prediction_data)
  
  temp_data = data_array[:, 0]
  temp_series = (temp_data / np.amax(temp_data)).reshape(len(data_array), 1)

  stress_intensity = Scaler.transform(data_array[:, 1].reshape(len(data_array), 1))

  return np.hstack((temp_series, stress_intensity))

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern. copy in groups?
		seq_x, seq_y = sequences[i:end_ix, 1:], sequences[end_ix, 0]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def create_actual_data(file_name):
  file = pd.read_csv(file_name)
  data_array = np.asarray(file)
  crack_length = data_array[:, 1]
  return crack_length

"""# Loading Data"""

#Load previously trained model
model = keras.models.load_model("./Weights/Model_HT_5_21_21_TM")
print("Model Loaded")

#Load in the experimental data
#['./Training/HeatTreated/t150_4-12-21_R05_1800N_training.csv', './Training/HeatTreated/t150_4-13-21_R05_1700N_training.csv', 
#'./Training/HeatTreated/t150_4-15-21_R05_1600N_training.csv', './Training/HeatTreated/t250_4-21-21_R05_1800N_training.csv', 
#'./Training/HeatTreated/t250_4-7-21_R05_1700N_training.csv', './Training/HeatTreated/t250_4-8-21_R05_1600N_training.csv', 
#'./Training/HeatTreated/t35_4-19-21_R05_1800N_training.csv', './Training/HeatTreated/t35_4-2-21_R05_1700N_training.csv']

data_1600_35 = np.asarray(pd.read_csv('./Training/HeatTreated/hold/t35_4-3-21_R05_1600N_training.bak'))
data_1700_35 = np.asarray(pd.read_csv('./Training/HeatTreated/t35_4-2-21_R05_1700N_training.csv'))
data_1800_35 = np.asarray(pd.read_csv('./Training/HeatTreated/t35_4-19-21_R05_1800N_training.csv'))

data_1600_150 = np.asarray(pd.read_csv('./Training/HeatTreated/t150_4-15-21_R05_1600N_training.csv'))
data_1700_150 = np.asarray(pd.read_csv('./Training/HeatTreated/t150_4-13-21_R05_1700N_training.csv'))
data_1800_150 = np.asarray(pd.read_csv('./Training/HeatTreated/t150_4-12-21_R05_1800N_training.csv'))

data_1600_250 = np.asarray(pd.read_csv('./Training/HeatTreated/t250_4-8-21_R05_1600N_training.csv'))
data_1700_250 = np.asarray(pd.read_csv('./Training/HeatTreated/t250_4-7-21_R05_1700N_training.csv'))
data_1800_250 = np.asarray(pd.read_csv('./Training/HeatTreated/t250_4-21-21_R05_1800N_training.csv'))

#Load in the extrapolated data
w_noise = 2 # load data with guassian noise 0=no 1=yes
if w_noise > 1:
  #75C
  data_1600_75 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_348K_noise.csv'))
  data_1700_75 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_348K_noise.csv'))
  data_1800_75 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_348K_noise.csv'))
  #200C
  data_1600_200 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_473K_noise.csv'))
  data_1700_200 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_473K_noise.csv'))
  data_1800_200 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_473K_noise.csv'))
  #300C
  data_1600_300 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_573K_noise.csv'))
  data_1700_300 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_573K_noise.csv'))
  data_1800_300 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_573K_noise.csv'))
  #400C
  data_1600_400 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_673K_noise.csv'))
  data_1700_400 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_673K_noise.csv'))
  data_1800_400 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_673K_noise.csv'))
  #500C
  data_1600_500 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_773K_noise.csv'))
  data_1700_500 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_773K_noise.csv'))
  data_1800_500 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_773K_noise.csv'))
  print('Data Loaded')
else:
  #75C
  data_1600_75 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_348K.csv'))
  data_1700_75 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_348K.csv'))
  data_1800_75 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_348K.csv'))
  #200C
  data_1600_200 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_473K.csv'))
  data_1700_200 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_473K.csv'))
  data_1800_200 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_473K.csv'))
  #300C
  data_1600_300 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_573K.csv'))
  data_1700_300 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_573K.csv'))
  data_1800_300 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_573K.csv'))
  #400C
  data_1600_400 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_673K.csv'))
  data_1700_400 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_673K.csv'))
  data_1800_400 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_673K.csv'))
  #500C
  data_1600_500 = np.asarray(pd.read_csv('./Predict/HeatTreated/1600N_HT_773K.csv'))
  data_1700_500 = np.asarray(pd.read_csv('./Predict/HeatTreated/1700N_HT_773K.csv'))
  data_1800_500 = np.asarray(pd.read_csv('./Predict/HeatTreated/1800N_HT_773K.csv'))

#this function computes how to offset the initial crackheight. should be set to zero.
def auto_crackoffset(crack_series):
  crack_avg = 0
  crack_offset = 0
  num_avg = 100
  #for i in range(0,num_avg):
  #  crack_avg += crack_series[i]
  crack_offset = np.max(crack_series[0:100]) #ended up offsetting my maximum value in the first 100 entries.
  return -crack_offset

#Read in the DATA - !!WARNING NEEDS TO BE CONSISTENT WITH TRAINING DATA!!

#Offset Crack to Zero by reading first 100 crack entries and offsetting to zero
offset_crack = 1 # 0=no 1=yes

# time
time_1600_35 = data_1600_35[:,0];time_1700_35 = data_1700_35[:,0];time_1800_35 = data_1800_35[:,0]
time_1600_150 = data_1600_150[:,0];time_1700_150 = data_1700_150[:,0];time_1800_150 = data_1800_150[:,0]
time_1600_250 = data_1600_250[:,0];time_1700_250 = data_1700_250[:,0];time_1800_250 = data_1800_250[:,0]
#extrapolated data
time_1600_75 = data_1600_75[:,0];time_1700_75 = data_1700_75[:,0];time_1800_75 = data_1800_75[:,0]
time_1600_200 = data_1600_200[:,0];time_1700_200 = data_1700_200[:,0];time_1800_200 = data_1800_200[:,0]
time_1600_300 = data_1600_300[:,0];time_1700_300 = data_1700_300[:,0];time_1800_300 = data_1800_300[:,0]
time_1600_400 = data_1600_400[:,0];time_1700_400 = data_1700_400[:,0];time_1800_400 = data_1800_400[:,0]
time_1600_500 = data_1600_500[:,0];time_1700_500 = data_1700_500[:,0];time_1800_500 = data_1800_500[:,0]

# crack length
if offset_crack > 0:
  crack_1600_35  = data_1600_35[:,1] +auto_crackoffset(data_1600_35[:,1]);crack_1700_35 = data_1700_35[:,1]+auto_crackoffset(data_1700_35[:,1]);crack_1800_35 = data_1800_35[:,1]+auto_crackoffset(data_1800_35[:,1])
  crack_1600_150 = data_1600_150[:,1]+auto_crackoffset(data_1600_150[:,1]);crack_1700_150 = data_1700_150[:,1]+auto_crackoffset(data_1700_150[:,1]);crack_1800_150 = data_1800_150[:,1]+auto_crackoffset(data_1800_150[:,1])
  crack_1600_250 = data_1600_250[:,1]+auto_crackoffset(data_1600_250[:,1]);crack_1700_250 = data_1700_250[:,1]+auto_crackoffset(data_1700_250[:,1]);crack_1800_250 = data_1800_250[:,1]+auto_crackoffset(data_1800_250[:,1])
else:
  crack_1600_35  = data_1600_35[:,1];crack_1700_35 = data_1700_35[:,1];crack_1800_35 = data_1800_35[:,1]
  crack_1600_150 = data_1600_150[:,1];crack_1700_150 = data_1700_150[:,1];crack_1800_150 = data_1800_150[:,1]
  crack_1600_250 = data_1600_250[:,1];crack_1700_250 = data_1700_250[:,1];crack_1800_250 = data_1800_250[:,1]
#extrapolated data
crack_1600_75 = data_1600_75[:,1];crack_1700_75 = data_1700_75[:,1];crack_1800_75 = data_1800_75[:,1]
crack_1600_200 = data_1600_200[:,1];crack_1700_200 = data_1700_200[:,1];crack_1800_200 = data_1800_200[:,1]
crack_1600_300 = data_1600_300[:,1];crack_1700_300 = data_1700_300[:,1];crack_1800_300 = data_1800_300[:,1]
crack_1600_400 = data_1600_400[:,1];crack_1700_400 = data_1700_400[:,1];crack_1800_400 = data_1800_400[:,1]
crack_1600_500 = data_1600_500[:,1];crack_1700_500 = data_1700_500[:,1];crack_1800_500 = data_1800_500[:,1]

# force
force_1600_35 = data_1600_35[:,2];force_1700_35 = data_1700_35[:,2];force_1800_35 = data_1800_35[:,2]
force_1600_150 = data_1600_150[:,2];force_1700_150 = data_1700_150[:,2];force_1800_150 = data_1800_150[:,2]
force_1600_250 = data_1600_250[:,2];force_1700_250 = data_1700_250[:,2];force_1800_250 = data_1800_250[:,2]
#extrapolated data
force_1600_75 = data_1600_75[:,2];force_1700_75 = data_1700_75[:,2];force_1800_75 = data_1800_75[:,2]
force_1600_200 = data_1600_200[:,2];force_1700_200 = data_1700_200[:,2];force_1800_200 = data_1800_200[:,2]
force_1600_300 = data_1600_300[:,2];force_1700_300 = data_1700_300[:,2];force_1800_300 = data_1800_300[:,2]
force_1600_400 = data_1600_400[:,2];force_1700_400 = data_1700_400[:,2];force_1800_400 = data_1800_400[:,2]
force_1600_500 = data_1600_500[:,2];force_1700_500 = data_1700_500[:,2];force_1800_500 = data_1800_500[:,2]

# normalize the temperature series data by 700K
temp_1600_35 = data_1600_35[:,3]/700;temp_1700_35 = data_1700_35[:,3]/700;temp_1800_35 = data_1800_35[:,3]/700
temp_1600_150 = data_1600_150[:,3]/700;temp_1700_150 = data_1700_150[:,3]/700;temp_1800_150 = data_1800_150[:,3]/700
temp_1600_250 = data_1600_250[:,3]/700;temp_1700_250 = data_1700_250[:,3]/700;temp_1800_250 = data_1800_250[:,3]/700
#extrapolated data
temp_1600_75 = data_1600_300[:,3]/700;temp_1700_75 = data_1700_75[:,3]/700;temp_1800_75 = data_1800_75[:,3]/700
temp_1600_200 = data_1600_200[:,3]/700;temp_1700_200 = data_1700_200[:,3]/700;temp_1800_200 = data_1800_200[:,3]/700
temp_1600_300 = data_1600_300[:,3]/700;temp_1700_300 = data_1700_300[:,3]/700;temp_1800_300 = data_1800_300[:,3]/700
temp_1600_400 = data_1600_400[:,3]/700;temp_1700_400 = data_1700_400[:,3]/700;temp_1800_400 = data_1800_400[:,3]/700
temp_1600_500 = data_1600_500[:,3]/700;temp_1700_500 = data_1700_500[:,3]/700;temp_1800_500 = data_1800_500[:,3]/700

#let's plot the data to see what is going on
fig, (ax1c1,ax2c1,ax3c1) = plt.subplots(3,1, sharex=True)
#Column 1
ax1c1.plot(time_1600_35[:,],crack_1600_35[:,], '-', label='35C 1600N') # 
ax1c1.plot(time_1700_35[:,],crack_1700_35[:,], '-', label='35C 1700N') # 
ax1c1.plot(time_1800_35[:,],crack_1800_35[:,], '-', label='35C 1800N') # 
ax2c1.plot(time_1600_35[:,],force_1600_35[:,], '-', label='35C 1600N')
ax2c1.plot(time_1700_35[:,],force_1700_35[:,], '-', label='35C 1700N')
ax2c1.plot(time_1800_35[:,],force_1800_35[:,], '-', label='35C 1800N')
ax3c1.plot(time_1600_35[:,],temp_1600_35[:,], '-', label='35C 1600N')
ax3c1.plot(time_1700_35[:,],temp_1700_35[:,], '-', label='35C 1700N')
ax3c1.plot(time_1800_35[:,],temp_1800_35[:,], '-', label='35C 1800N')

fig2, (ax1c2,ax2c2,ax3c2) = plt.subplots(3,1, sharex=True)
#Column 2
ax1c2.plot(time_1600_150[:,],crack_1600_150[:,], '-', label='150C 1600N') # 
ax1c2.plot(time_1700_150[:,],crack_1700_150[:,], '-', label='150C 1700N') # 
ax1c2.plot(time_1800_150[:,],crack_1800_150[:,], '-', label='150C 1800N') # 
ax2c2.plot(time_1600_150[:,],force_1600_150[:,], '-', label='150C 1600N')
ax2c2.plot(time_1700_150[:,],force_1700_150[:,], '-', label='150C 1700N')
ax2c2.plot(time_1800_150[:,],force_1800_150[:,], '-', label='150C 1800N')
ax3c2.plot(time_1600_150[:,],temp_1600_150[:,], '-', label='150C 1600N')
ax3c2.plot(time_1700_150[:,],temp_1700_150[:,], '-', label='150C 1700N')
ax3c2.plot(time_1800_150[:,],temp_1800_150[:,], '-', label='150C 1800N')

fig3, (ax1c3,ax2c3,ax3c3) = plt.subplots(3,1, sharex=True)
#Column 3
ax1c3.plot(time_1600_250[:,],crack_1600_250[:,], '-', label='250C 1600N') # 
ax1c3.plot(time_1700_250[:,],crack_1700_250[:,], '-', label='250C 1700N') # 
ax1c3.plot(time_1800_250[:,],crack_1800_250[:,], '-', label='250C 1800N') # 
ax2c3.plot(time_1600_250[:,],force_1600_250[:,], '-', label='250C 1600N')
ax2c3.plot(time_1700_250[:,],force_1700_250[:,], '-', label='250C 1700N')
ax2c3.plot(time_1800_250[:,],force_1800_250[:,], '-', label='250C 1800N')
ax3c3.plot(time_1600_250[:,],temp_1600_250[:,], '-', label='250C 1600N')
ax3c3.plot(time_1700_250[:,],temp_1700_250[:,], '-', label='250C 1700N')
ax3c3.plot(time_1800_250[:,],temp_1800_250[:,], '-', label='250C 1800N')
plt.xlabel('Cycle Number')
plt.show()

fig4, (ax1c4,ax2c4,ax3c4) = plt.subplots(3,1, sharex=True)
#Column 3
ax1c4.plot(time_1600_300[:,],crack_1600_300[:,], '-', label='300C 1600N') # 
ax1c4.plot(time_1700_300[:,],crack_1700_300[:,], '-', label='300C 1700N') # 
ax1c4.plot(time_1800_300[:,],crack_1800_300[:,], '-', label='300C 1800N') # 
ax2c4.plot(time_1600_300[:,],force_1600_300[:,], '-', label='300C 1600N')
ax2c4.plot(time_1700_300[:,],force_1700_300[:,], '-', label='300C 1700N')
ax2c4.plot(time_1800_300[:,],force_1800_300[:,], '-', label='300C 1800N')
ax3c4.plot(time_1600_300[:,],temp_1600_300[:,], '-', label='300C 1600N')
ax3c4.plot(time_1700_300[:,],temp_1700_300[:,], '-', label='300C 1700N')
ax3c4.plot(time_1800_300[:,],temp_1800_300[:,], '-', label='300C 1800N')
plt.xlabel('Cycle Number')
plt.show()

#so it looks like it is concatenated 1800,1700,1600(med temperature) then 1800,1700,1600(high temperature) then 1800,1700,1600 (room temperature)
#don't quite understand the order or reasoning. but let's see if it works.

#funtion to compute time step. we must maintain time step as we used when we trained the data
def avg_timestep(time_series):
  diff = 0
  w=0
  for i in range(0,len(time_series)-1):
    diff += time_series[i+1,]-time_series[i,]
    if time_series[i+1,]-time_series[i,]>0.196 and w<3:
      print('Warning!! Timestep Change > 0.196',i,time_series[i+1,]-time_series[i,])
      w+=1

  return diff/len(time_series)

#funtion creates a timestep array
def timestep_arr(time_series):
  diff = 0
  ts = []
  for i in range(0,len(time_series)-1):
    diff = time_series[i+1,]-time_series[i,]
    ts.append(diff)
  return np.array(ts)

#let's check the time
#check time step
tstep_1600_35 = timestep_arr(time_1600_35)
tstep_1600_150 = timestep_arr(time_1600_150)
tstep_1600_200 = timestep_arr(time_1600_250)
#extrapolated
tstep_1600_75 = timestep_arr(time_1600_75)
tstep_1600_200 = timestep_arr(time_1600_200)
tstep_1600_300 = timestep_arr(time_1600_300)
tstep_1600_200 = timestep_arr(time_1600_400)
tstep_1600_300 = timestep_arr(time_1600_500)

print('Average Time Sampling - 35C',avg_timestep(time_1600_35))
print('Average Time Sampling - 150C',avg_timestep(time_1600_150))
print('Average Time Sampling - 250C',avg_timestep(time_1600_250))
print('Average Time Sampling - 300C',avg_timestep(time_1600_300))

#
fig5, (ax1c4) = plt.subplots(1,1, sharex=True)
#Column 3
ax1c4.plot(tstep_1600_35[:,], '-', label='35C 1600N') # 
ax1c4.plot(tstep_1600_150[:,], '-', label='150C 1600N') # 
ax1c4.plot(tstep_1600_200[:,], '-', label='200C 1600N') # 
ax1c4.plot(tstep_1600_300[:,], '-', label='300C 1600N') # 
ax1c4.set_xlim([1450, 1600])
plt.xlabel('Cycle Number')
plt.show()

#Function to calculate the stress intensity.
# we are being a little lazy here. Should create header file with these common functions.
#
#  !!!!!!! WARNING CHECK TO MAKE SURE SAME AS TRAINING FUNCTION  !!!!!!!!!!!
#
# Note: we offset by the initial crack stress intensity!!!!
#
def calc_stress_intensity(force_series,crack_series,a0_off):
  #Calculate the stress intensity
  #the geometry of the specimen had to be updated on 5/19/21
  stress_intensity = []
  intensity0 = 0
  #
  # Sample Parameters
  specimen_width = 34.63 #meters #40.8 mm new width 34.63mm
  thickness = 0.4572 #meters thickness of sample 0.4572mm
  a0 = 10 #initial crack length is 10mm
  crossSectArea = specimen_width*thickness #mm^2 to get MPa

  #looks like you are using the polynomial method derived from FEA to determine stress intensity
  for i in range(len(crack_series)):
    if crack_series[i]>0:
      if a0_off>0:
        #this first stress intensity is for the initial crack only
        term1 = (force_series[i] / crossSectArea) * sqrt(3.14159 * (a0)) #N/mm^2*sqrt(mm) = MPa*sqrt(mm)
        term2 = 1.122
        term3 = -0.231 * ((a0)  / specimen_width)
        term4 = 10.55 * (((a0) / specimen_width)**2)
        term5 = -21.71 * (((a0) / specimen_width)**3)
        term6 = 30.382 * (((a0) / specimen_width)**4)
        intensity0 = term1*(term2 + term3 + term4 + term5 + term6) #initial crack really only need to calculate for each load but wtf
      else:
        intensity0 = 0

      term1 = (force_series[i] / crossSectArea) * sqrt(3.14159 * (crack_series[i]+a0)) #N/mm^2*sqrt(mm) = MPa*sqrt(mm)
      term2 = 1.122
      term3 = -0.231 * ((crack_series[i]+a0)  / specimen_width)
      term4 = 10.55 * (((crack_series[i]+a0) / specimen_width)**2)
      term5 = -21.71 * (((crack_series[i]+a0) / specimen_width)**3)
      term6 = 30.382 * (((crack_series[i]+a0) / specimen_width)**4)
      intensity = term1*(term2 + term3 + term4 + term5 + term6)-intensity0 # basically shift SI down by initial crack value

    else:
      intensity = 0
    stress_intensity.append(intensity)
    if i % 100000 == 0:
      print(i)
  return np.array(stress_intensity)

#Read Stress Intesnity Scaler from Training
Scaler = MinMaxScaler()
Scaler = joblib.load('./Weights/Scaler_HT_TM.save')
print(Scaler.get_params)

#Calculate the stress intensity - 35C
si_1600_35_bt = calc_stress_intensity(force_1600_35,crack_1600_35,1)
si_1700_35_bt = calc_stress_intensity(force_1700_35,crack_1700_35,1)
si_1800_35_bt = calc_stress_intensity(force_1800_35,crack_1800_35,1)

#Calculate the stress intensity - 150C
si_1600_150_bt = calc_stress_intensity(force_1600_150,crack_1600_150,1)
si_1700_150_bt = calc_stress_intensity(force_1700_150,crack_1700_150,1)
si_1800_150_bt = calc_stress_intensity(force_1800_150,crack_1800_150,1)

#Calculate the stress intensity - 150C
si_1600_250_bt = calc_stress_intensity(force_1600_250,crack_1600_250,1)
si_1700_250_bt = calc_stress_intensity(force_1700_250,crack_1700_250,1)
si_1800_250_bt = calc_stress_intensity(force_1800_250,crack_1800_250,1)
#Extrapolated Data
#Calculate the stress intensity - 75C
si_1600_75_bt = calc_stress_intensity(force_1600_75,crack_1600_75,1)
si_1700_75_bt = calc_stress_intensity(force_1700_75,crack_1700_75,1)
si_1800_75_bt = calc_stress_intensity(force_1800_75,crack_1800_75,1)

#Calculate the stress intensity - 200C
si_1600_200_bt = calc_stress_intensity(force_1600_200,crack_1600_200,1)
si_1700_200_bt = calc_stress_intensity(force_1700_200,crack_1700_200,1)
si_1800_200_bt = calc_stress_intensity(force_1800_200,crack_1800_200,1)

#Calculate the stress intensity - 300C
si_1600_300_bt = calc_stress_intensity(force_1600_300,crack_1600_300,1)
si_1700_300_bt = calc_stress_intensity(force_1700_300,crack_1700_300,1)
si_1800_300_bt = calc_stress_intensity(force_1800_300,crack_1800_300,1)

#Calculate the stress intensity - 400C
si_1600_400_bt = calc_stress_intensity(force_1600_400,crack_1600_400,1)
si_1700_400_bt = calc_stress_intensity(force_1700_400,crack_1700_400,1)
si_1800_400_bt = calc_stress_intensity(force_1800_400,crack_1800_400,1)

#Calculate the stress intensity - 500C
si_1600_500_bt = calc_stress_intensity(force_1600_500,crack_1600_500,1)
si_1700_500_bt = calc_stress_intensity(force_1700_500,crack_1700_500,1)
si_1800_500_bt = calc_stress_intensity(force_1800_500,crack_1800_500,1)


#plot before scaling
fig2, (ax1c1) = plt.subplots(1,1, sharex=True)
#Column 1
#ax1c1.plot(time_1600_35[:,],si_1600_35[:,], 'r-', label='35C 1600N') # 
#ax1c1.plot(time_1700_35[:,],si_1700_35[:,], 'g-', label='35C 1700N') # 
#ax1c1.plot(time_1800_35[:,],si_1800_35[:,], 'b-', label='35C 1800N') #

#ax1c1.plot(time_1600_150[:,],si_1600_150[:,], 'r-', label='150C 1600N') # 
#ax1c1.plot(time_1700_150[:,],si_1700_150[:,], 'g-', label='150C 1700N') # 
#ax1c1.plot(time_1800_150[:,],si_1800_150[:,], 'b-', label='150C 1800N') # 

ax1c1.plot(time_1600_250[:,],si_1600_250_bt[:,], 'r-', label='250C 1600N') # 
ax1c1.plot(time_1700_250[:,],si_1700_250_bt[:,], 'g-', label='250C 1700N') # 
ax1c1.plot(time_1800_250[:,],si_1800_250_bt[:,], 'b-', label='250C 1800N') # 

ax1c1.plot(time_1600_300[:,],si_1600_300_bt[:,], 'r-', label='300C 1600N Expl') # 
ax1c1.plot(time_1700_300[:,],si_1700_300_bt[:,], 'g-', label='300C 1700N Expl') # 
ax1c1.plot(time_1800_300[:,],si_1800_300_bt[:,], 'b-', label='300C 1800N Expl') # 

#scale stress intensity by same factor used during Training
si_1600_35 = Scaler.transform(si_1600_35_bt.reshape((len(si_1600_35_bt),1)))
si_1700_35 = Scaler.transform(si_1700_35_bt.reshape((len(si_1700_35_bt),1)))
si_1800_35 = Scaler.transform(si_1800_35_bt.reshape((len(si_1800_35_bt),1)))
#scale stress intensity by same factor used during Training
si_1600_150 = Scaler.transform(si_1600_150_bt.reshape((len(si_1600_150_bt),1)))
si_1700_150 = Scaler.transform(si_1700_150_bt.reshape((len(si_1700_150_bt),1)))
si_1800_150 = Scaler.transform(si_1800_150_bt.reshape((len(si_1800_150_bt),1)))
#scale stress intensity by same factor used during Training
si_1600_250 = Scaler.transform(si_1600_250_bt.reshape((len(si_1600_250_bt),1)))
si_1700_250 = Scaler.transform(si_1700_250_bt.reshape((len(si_1700_250_bt),1)))
si_1800_250 = Scaler.transform(si_1800_250_bt.reshape((len(si_1800_250_bt),1)))
#Extrapolated
#75C
si_1600_75 = Scaler.transform(si_1600_75_bt.reshape((len(si_1600_75_bt),1)))
si_1700_75 = Scaler.transform(si_1700_75_bt.reshape((len(si_1700_75_bt),1)))
si_1800_75 = Scaler.transform(si_1800_75_bt.reshape((len(si_1800_75_bt),1)))
#200C
si_1600_200 = Scaler.transform(si_1600_200_bt.reshape((len(si_1600_200_bt),1)))
si_1700_200 = Scaler.transform(si_1700_200_bt.reshape((len(si_1700_200_bt),1)))
si_1800_200 = Scaler.transform(si_1800_200_bt.reshape((len(si_1800_200_bt),1)))
#300C
si_1600_300 = Scaler.transform(si_1600_300_bt.reshape((len(si_1600_300_bt),1)))
si_1700_300 = Scaler.transform(si_1700_300_bt.reshape((len(si_1700_300_bt),1)))
si_1800_300 = Scaler.transform(si_1800_300_bt.reshape((len(si_1800_300_bt),1)))
#400C
si_1600_400 = Scaler.transform(si_1600_400_bt.reshape((len(si_1600_400_bt),1)))
si_1700_400 = Scaler.transform(si_1700_400_bt.reshape((len(si_1700_400_bt),1)))
si_1800_400 = Scaler.transform(si_1800_400_bt.reshape((len(si_1800_400_bt),1)))
#500C
si_1600_500 = Scaler.transform(si_1600_500_bt.reshape((len(si_1600_500_bt),1)))
si_1700_500 = Scaler.transform(si_1700_500_bt.reshape((len(si_1700_500_bt),1)))
si_1800_500 = Scaler.transform(si_1800_500_bt.reshape((len(si_1800_500_bt),1)))
#plot after scaling
fig2, (ax1c1) = plt.subplots(1,1, sharex=True)
#Column 1
#ax1c1.plot(time_1600_35[:,],si_1600_35[:,], 'r-', label='35C 1600N') # 
#ax1c1.plot(time_1700_35[:,],si_1700_35[:,], 'g-', label='35C 1700N') # 
#ax1c1.plot(time_1800_35[:,],si_1800_35[:,], 'b-', label='35C 1800N') #

#ax1c1.plot(time_1600_150[:,],si_1600_150[:,], 'r-', label='150C 1600N') # 
#ax1c1.plot(time_1700_150[:,],si_1700_150[:,], 'g-', label='150C 1700N') # 
#ax1c1.plot(time_1800_150[:,],si_1800_150[:,], 'b-', label='150C 1800N') # 

ax1c1.plot(time_1600_250[:,],si_1600_250[:,], 'r-', label='250C 1600N') # 
ax1c1.plot(time_1700_250[:,],si_1700_250[:,], 'g-', label='250C 1700N') # 
ax1c1.plot(time_1800_250[:,],si_1800_250[:,], 'b-', label='250C 1800N') # 

ax1c1.plot(time_1600_300[:,],si_1600_300[:,], 'r-', label='300C 1600N Expl') # 
ax1c1.plot(time_1700_300[:,],si_1700_300[:,], 'g-', label='300C 1700N Expl') # 
ax1c1.plot(time_1800_300[:,],si_1800_300[:,], 'b-', label='300C 1800N Expl') #

#stack the data 
data_1600_35 = np.vstack((crack_1600_35, temp_1600_35, si_1600_35.reshape((len(si_1600_35),)))).swapaxes(0,1) #horizontal stack data
data_1700_35 = np.vstack((crack_1700_35, temp_1700_35, si_1700_35.reshape((len(si_1700_35),)))).swapaxes(0,1) #horizontal stack data
data_1800_35 = np.vstack((crack_1800_35, temp_1800_35, si_1800_35.reshape((len(si_1800_35),)))).swapaxes(0,1) #horizontal stack data

data_1600_150 = np.vstack((crack_1600_150, temp_1600_150, si_1600_150.reshape((len(si_1600_150),)))).swapaxes(0,1) #horizontal stack data
data_1700_150 = np.vstack((crack_1700_150, temp_1700_150, si_1700_150.reshape((len(si_1700_150),)))).swapaxes(0,1) #horizontal stack data
data_1800_150 = np.vstack((crack_1800_150, temp_1800_150, si_1800_150.reshape((len(si_1800_150),)))).swapaxes(0,1) #horizontal stack data

data_1600_250 = np.vstack((crack_1600_250, temp_1600_250, si_1600_250.reshape((len(si_1600_250),)))).swapaxes(0,1) #horizontal stack data
data_1700_250 = np.vstack((crack_1700_250, temp_1700_250, si_1700_250.reshape((len(si_1700_250),)))).swapaxes(0,1) #horizontal stack data
data_1800_250 = np.vstack((crack_1800_250, temp_1800_250, si_1800_250.reshape((len(si_1800_250),)))).swapaxes(0,1) #horizontal stack data
#extrapolated
#75C
data_1600_75 = np.vstack((crack_1600_75, temp_1600_75, si_1600_75.reshape((len(si_1600_75),)))).swapaxes(0,1) #horizontal stack data
data_1700_75 = np.vstack((crack_1700_75, temp_1700_75, si_1700_75.reshape((len(si_1700_75),)))).swapaxes(0,1) #horizontal stack data
data_1800_75 = np.vstack((crack_1800_75, temp_1800_75, si_1800_75.reshape((len(si_1800_75),)))).swapaxes(0,1) #horizontal stack data
#200C
data_1600_200 = np.vstack((crack_1600_200, temp_1600_200, si_1600_200.reshape((len(si_1600_200),)))).swapaxes(0,1) #horizontal stack data
data_1700_200 = np.vstack((crack_1700_200, temp_1700_200, si_1700_200.reshape((len(si_1700_200),)))).swapaxes(0,1) #horizontal stack data
data_1800_200 = np.vstack((crack_1800_200, temp_1800_200, si_1800_200.reshape((len(si_1800_200),)))).swapaxes(0,1) #horizontal stack data
#300C
data_1600_300 = np.vstack((crack_1600_300, temp_1600_300, si_1600_300.reshape((len(si_1600_300),)))).swapaxes(0,1) #horizontal stack data
data_1700_300 = np.vstack((crack_1700_300, temp_1700_300, si_1700_300.reshape((len(si_1700_300),)))).swapaxes(0,1) #horizontal stack data
data_1800_300 = np.vstack((crack_1800_300, temp_1800_300, si_1800_300.reshape((len(si_1800_300),)))).swapaxes(0,1) #horizontal stack data
#400C
data_1600_400 = np.vstack((crack_1600_400, temp_1600_400, si_1600_400.reshape((len(si_1600_400),)))).swapaxes(0,1) #horizontal stack data
data_1700_400 = np.vstack((crack_1700_400, temp_1700_400, si_1700_400.reshape((len(si_1700_400),)))).swapaxes(0,1) #horizontal stack data
data_1800_400 = np.vstack((crack_1800_400, temp_1800_400, si_1800_400.reshape((len(si_1800_400),)))).swapaxes(0,1) #horizontal stack data
#500C
data_1600_500 = np.vstack((crack_1600_500, temp_1600_500, si_1600_500.reshape((len(si_1600_500),)))).swapaxes(0,1) #horizontal stack data
data_1700_500 = np.vstack((crack_1700_500, temp_1700_500, si_1700_500.reshape((len(si_1700_500),)))).swapaxes(0,1) #horizontal stack data
data_1800_500 = np.vstack((crack_1800_500, temp_1800_500, si_1800_500.reshape((len(si_1800_500),)))).swapaxes(0,1) #horizontal stack data

"""# Create Prediction Data"""

#split the sequences. [i1 i2 i3] [o4]
# temperature and stress intensity are inputs
n_steps = 3

xsplit_1600_35, ysplit_1600_35 = split_sequences(data_1600_35, n_steps)
xsplit_1700_35, ysplit_1700_35 = split_sequences(data_1700_35, n_steps)
xsplit_1800_35, ysplit_1800_35 = split_sequences(data_1800_35, n_steps)

xsplit_1600_150, ysplit_1600_150 = split_sequences(data_1600_150, n_steps)
xsplit_1700_150, ysplit_1700_150 = split_sequences(data_1700_150, n_steps)
xsplit_1800_150, ysplit_1800_150 = split_sequences(data_1800_150, n_steps)

xsplit_1600_250, ysplit_1600_250 = split_sequences(data_1600_250, n_steps)
xsplit_1700_250, ysplit_1700_250 = split_sequences(data_1700_250, n_steps)
xsplit_1800_250, ysplit_1800_250 = split_sequences(data_1800_250, n_steps)
#Extrapolated
#75C
xsplit_1600_75, ysplit_1600_75 = split_sequences(data_1600_75, n_steps)
xsplit_1700_75, ysplit_1700_75 = split_sequences(data_1700_75, n_steps)
xsplit_1800_75, ysplit_1800_75 = split_sequences(data_1800_75, n_steps)
#200C
xsplit_1600_200, ysplit_1600_200 = split_sequences(data_1600_200, n_steps)
xsplit_1700_200, ysplit_1700_200 = split_sequences(data_1700_200, n_steps)
xsplit_1800_200, ysplit_1800_200 = split_sequences(data_1800_200, n_steps)
#300C
xsplit_1600_300, ysplit_1600_300 = split_sequences(data_1600_300, n_steps)
xsplit_1700_300, ysplit_1700_300 = split_sequences(data_1700_300, n_steps)
xsplit_1800_300, ysplit_1800_300 = split_sequences(data_1800_300, n_steps)
#400C
xsplit_1600_400, ysplit_1600_400 = split_sequences(data_1600_400, n_steps)
xsplit_1700_400, ysplit_1700_400 = split_sequences(data_1700_400, n_steps)
xsplit_1800_400, ysplit_1800_400 = split_sequences(data_1800_400, n_steps)
#500C
xsplit_1600_500, ysplit_1600_500 = split_sequences(data_1600_500, n_steps)
xsplit_1700_500, ysplit_1700_500 = split_sequences(data_1700_500, n_steps)
xsplit_1800_500, ysplit_1800_500 = split_sequences(data_1800_500, n_steps)

fig, (ax,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,sharey=True)
ax.plot(xsplit_1600_35[:,0,1], 'r--', label='1600N 35C') # 
ax.plot(xsplit_1700_35[:,0,1], 'g--', label='1700N 35C')
ax.plot(xsplit_1800_35[:,0,1], 'b--', label='1800N 35C')
ax2.plot(xsplit_1600_150[:,0,1], 'r-', label='1600N 150C') # 
ax2.plot(xsplit_1700_150[:,0,1], 'g--', label='1700N 150C')
ax2.plot(xsplit_1800_150[:,0,1], 'b--', label='1800N 150C')
ax3.plot(xsplit_1600_250[:,0,1], 'r--', label='1600N 250C') # 
ax3.plot(xsplit_1700_250[:,0,1], 'g--', label='1700N 250C')
ax3.plot(xsplit_1800_250[:,0,1], 'b--', label='1800N 250C')
ax4.plot(xsplit_1600_300[:,0,1], 'r--', label='1600N 300C') # 
ax4.plot(xsplit_1700_300[:,0,1], 'g--', label='1700N 300C')
ax4.plot(xsplit_1800_300[:,0,1], 'b--', label='1800N 300C')
#ax.legend()
plt.show()

"""# Predict On Data"""

# Predict the data using the trained model.
yhat1600_35 = model.predict(xsplit_1600_35, verbose=1)
yhat1700_35 = model.predict(xsplit_1700_35, verbose=1)
yhat1800_35 = model.predict(xsplit_1800_35, verbose=1)
print('35C FINISHED')

yhat1600_150 = model.predict(xsplit_1600_150, verbose=1)
yhat1700_150 = model.predict(xsplit_1700_150, verbose=1)
yhat1800_150 = model.predict(xsplit_1800_150, verbose=1)
print('150C FINISHED')

yhat1600_250 = model.predict(xsplit_1600_250, verbose=1)
yhat1700_250 = model.predict(xsplit_1700_250, verbose=1)
yhat1800_250 = model.predict(xsplit_1800_250, verbose=1)
print('250C FINISHED')

#Extrapolated Data
#75C
yhat1600_75 = model.predict(xsplit_1600_75, verbose=1)
yhat1700_75 = model.predict(xsplit_1700_75, verbose=1)
yhat1800_75 = model.predict(xsplit_1800_75, verbose=1)
print('300C FINISHED')
#200C
yhat1600_200 = model.predict(xsplit_1600_200, verbose=1)
yhat1700_200 = model.predict(xsplit_1700_200, verbose=1)
yhat1800_200 = model.predict(xsplit_1800_200, verbose=1)
print('300C FINISHED')
#300C
yhat1600_300 = model.predict(xsplit_1600_300, verbose=1)
yhat1700_300 = model.predict(xsplit_1700_300, verbose=1)
yhat1800_300 = model.predict(xsplit_1800_300, verbose=1)
print('300C FINISHED')
#400C
yhat1600_400 = model.predict(xsplit_1600_400, verbose=1)
yhat1700_400 = model.predict(xsplit_1700_400, verbose=1)
yhat1800_400 = model.predict(xsplit_1800_400, verbose=1)
print('300C FINISHED')
#500C
yhat1600_500 = model.predict(xsplit_1600_500, verbose=1)
yhat1700_500 = model.predict(xsplit_1700_500, verbose=1)
yhat1800_500 = model.predict(xsplit_1800_500, verbose=1)
print('300C FINISHED')

#Scale the Crack Series if it was Scaled During Training
# The crack length in millimeters is less than 3 so we likely will not need to scale.
scale_crackseries = 0 # 0=no 1=yes
if scale_crackseries > 0:
  Scaler_SC = MinMaxScaler()
  Scaler_SC = joblib.load('./Weights/Scaler_HT_TM_CS.save')
  print(Scaler.get_params)

  yhat1600_35 = Scaler_SC.inverse_transform(yhat1600_35)
  yhat1700_35 = Scaler_SC.inverse_transform(yhat1700_35)
  yhat1800_35 = Scaler_SC.inverse_transform(yhat1800_35)

  yhat1600_150 = Scaler_SC.inverse_transform(yhat1600_150)
  yhat1700_150 = Scaler_SC.inverse_transform(yhat1700_150)
  yhat1800_150 = Scaler_SC.inverse_transform(yhat1800_150)

  yhat1600_250 = Scaler_SC.inverse_transform(yhat1600_250)
  yhat1700_250 = Scaler_SC.inverse_transform(yhat1700_250)
  yhat1800_250 = Scaler_SC.inverse_transform(yhat1800_250)

"""# Plotting With Actual Data"""

#read data in again to plot against prediction

cl1600_35 = crack_1600_35
cl1700_35 = crack_1700_35
cl1800_35 = crack_1800_35

cl1600_150 = crack_1600_150
cl1700_150 = crack_1700_150
cl1800_150 = crack_1800_150

cl1600_250 = crack_1600_250
cl1700_250 = crack_1700_250
cl1800_250 = crack_1800_250
print(cl1800_250)

"""Plot against Original Data"""

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,20))

ax1.plot(cl1600_35, label = 'Exp-1600N@RT')
ax1.plot(cl1600_150, label = 'Exp-1600N@150C')
ax1.plot(cl1600_250, label = 'Exp-1600N@250C')
ax1.plot(yhat1600_35, label = 'ML-1600N@RT')
ax1.plot(yhat1600_150, label = 'ML-1600N@150C')
ax1.plot(yhat1600_250, label = 'ML-1600N@250C')
ax1.plot(yhat1600_300, label = 'ML-1600N@300C')
ax1.set_xlabel("Cycle Number")
ax1.set_ylabel("Crack Length (mm)")
ax1.set_ylim([0, 4])
ax1.legend(loc=2)
ax1.annotate('A)',xy=(575, 1100), xycoords='figure points', fontsize=25, color='Black')

ax2.plot(cl1700_35, label = 'Exp-1700N@RT')
ax2.plot(cl1700_150, label = 'Exp-1700N@150C')
ax2.plot(cl1700_250, label = 'Exp-1700N@250C')
ax2.plot(yhat1700_35, label = 'ML-1700N@RT')
ax2.plot(yhat1700_150, label = 'ML-1700N@150C')
ax2.plot(yhat1700_250, label = 'ML-1700N@250C')
ax2.plot(yhat1700_300, label = 'ML-1700N@300C')
ax2.set_xlabel("Cycle Number")
ax2.set_ylabel("Crack Length (mm)")
ax2.legend()
ax2.set_ylim([0, 4])
ax2.annotate('B)',xy=(575, 725), xycoords='figure points', fontsize=25, color='Black')

ax3.plot(cl1800_35, label = 'Exp-1800N@RT')
ax3.plot(cl1800_150, label = 'Exp-1800N@150C')
ax3.plot(cl1800_250, label = 'Exp-1800N@250C')
ax3.plot(yhat1800_35, label = 'ML-1800N@RT')
ax3.plot(yhat1800_150, label = 'ML-1800N@150C')
ax3.plot(yhat1800_250, label = 'ML-1800N@250C')
ax3.plot(yhat1800_300, label = 'ML-1800N@300C')
ax3.set_xlabel('Cycle Number')
ax3.set_ylabel('Crack Length (mm)')
ax3.legend()
ax3.set_ylim([0, 4])
ax3.annotate('C)',xy=(575, 340), xycoords='figure points', fontsize=25, color='Black')

#ax3.xlabel("Cycle Number")
#ax1.ylabel("Crack Length (mm)")

plt.savefig('./Paper Images/CrackLength_HT_vT_TM.pdf', format='pdf',bbox_inches='tight')

"""Paper Figure"""

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,20))

#150C Data
ax1.plot(cl1600_150, label = 'Original (1600N) @ 150\u2103')
ax1.plot(yhat1600_150, label = 'Predicted (1600N) @ 150\u2103')
ax1.plot(cl1700_150, label = 'Original (1700N) @ 150\u2103')
ax1.plot(yhat1700_150, label = 'Predicted (1700N) @ 150\u2103')
ax1.plot(cl1800_150, label = 'Original (1800N) @ 150\u2103')
ax1.plot(yhat1800_150, label = 'Predicted (1800N) @ 150\u2103')
ax1.set_xlabel("Sample Number")
ax1.set_ylabel("Crack Length (mm)")
ax1.set_ylim([0, 4])
ax1.legend(loc=2)
ax1.annotate('A) @150\u2103',xy=(475, 1100), xycoords='figure points', fontsize=25, color='Black')

#250C Data
ax2.plot(cl1600_250, label = 'Original (1800N)  @ 250\u2103')
ax2.plot(yhat1600_250, label = 'Predicted (1600N)  @ 250\u2103')
ax2.plot(cl1700_250, label = 'Original (1700N)  @ 250\u2103')
ax2.plot(yhat1700_250, label = 'Predicted (1700N)  @ 250\u2103')
ax2.plot(cl1800_250, label = 'Original (1800N)  @ 250\u2103')
ax2.plot(yhat1800_250, label = 'Predicted (1800N)  @ 250\u2103')
ax2.set_xlabel("Sample Number")
ax2.set_ylabel("Crack Length (mm)")
ax2.legend()
ax2.set_ylim([0, 4])
ax2.annotate('B) @250\u2103',xy=(475, 725), xycoords='figure points', fontsize=25, color='Black')

#300C Data
ax3.plot(yhat1600_300, label = 'Predicted (1600N)  @ 300\u2103')
ax3.plot(yhat1700_300, label = 'Predicted (1700N)  @ 300\u2103')
ax3.plot(yhat1800_300, label = 'Predicted (1800N)  @ 300\u2103')
ax3.set_xlabel('Sample Number')
ax3.set_ylabel('Crack Length (mm)')
ax3.legend()
ax3.set_ylim([0, 4])
ax3.annotate('C) @ 300\u2103',xy=(475, 340), xycoords='figure points', fontsize=25, color='Black')

#plt.savefig('./Paper Images/fig2.pdf', format='pdf',bbox_inches='tight')

print(yhat1800_150.shape)

"""Heatmap"""

#Assemble arrays for 30C
min_arr = np.amin([len(yhat1600_35),len(yhat1700_35),len(yhat1800_35)])
arr_force = np.ones(min_arr).reshape(min_arr,1)
force_heatmap_35 = np.hstack((arr_force*1600,arr_force*1700,arr_force*1800))
crack_heatmap_35 = np.hstack((yhat1600_35[:min_arr,],yhat1700_35[:min_arr,],yhat1800_35[:min_arr,]))

arr_ti = np.arange(0,min_arr).reshape(min_arr,1)
time_heatmap_35 = np.hstack((arr_ti,arr_ti,arr_ti))

#Assemble arrays for 250C
min_arr = np.amin([len(yhat1600_250),len(yhat1700_250),len(yhat1800_250)])
arr_force = np.ones(min_arr).reshape(min_arr,1)
force_heatmap_250 = np.hstack((arr_force*1600,arr_force*1700,arr_force*1800))
crack_heatmap_250 = np.hstack((yhat1600_250[:min_arr,],yhat1700_250[:min_arr,],yhat1800_250[:min_arr,]))

arr_ti = np.arange(0,min_arr).reshape(min_arr,1)
time_heatmap_250 = np.hstack((arr_ti,arr_ti,arr_ti))

#Assemble arrays for 300C
min_arr = np.amin([len(yhat1600_300),len(yhat1700_300),len(yhat1800_300)])
arr_force = np.ones(min_arr).reshape(min_arr,1)
force_heatmap_300 = np.hstack((arr_force*1600,arr_force*1700,arr_force*1800))
crack_heatmap_300 = np.hstack((yhat1600_300[:min_arr,],yhat1700_300[:min_arr,],yhat1800_300[:min_arr,]))

arr_ti = np.arange(0,min_arr).reshape(min_arr,1)
time_heatmap_300 = np.hstack((arr_ti,arr_ti,arr_ti))

opts = {'vmin': 0, 'vmax': 3, 'cmap': 'RdGy', 'levels':23}
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,15))

plot1 = ax1.contourf(force_heatmap_35,time_heatmap_35,crack_heatmap_35, **opts)
ax1.set_xlabel("Force (N)")
ax1.set_ylabel("Sample Number")
ax1.annotate('A) @20\u2103',xy=(0.65, 0.73), xycoords='figure fraction', fontsize=20, color='white')

plot2 = ax2.contourf(force_heatmap_250,time_heatmap_250,crack_heatmap_250, **opts)
ax2.set_xlabel("Force (N)")
ax2.set_ylabel("Sample Number")
ax2.annotate('B) @250\u2103',xy=(0.65, 0.41), xycoords='figure fraction', fontsize=20, color='white')

plot3 = ax3.contourf(force_heatmap_300,time_heatmap_300,crack_heatmap_300, **opts)
ax3.set_xlabel("Force (N)")
ax3.set_ylabel("Sample Number")
ax3.annotate('C) @300\u2103',xy=(0.65, 0.08), xycoords='figure fraction', fontsize=20, color='white')


cbar1 = fig.colorbar(plot1, ax=ax1)
cbar1.set_label('Crack Length (mm)')
cbar2 = fig.colorbar(plot2, ax=ax2)
cbar2.set_label('Crack Length (mm)')
cbar3 = fig.colorbar(plot3, ax=ax3)
cbar3.set_label('Crack Length (mm)')

#plt.savefig('./Paper Images/Heatmap_HT_TM_6_8_21.pdf', format='pdf',bbox_inches='tight')
#plt.savefig('./Paper Images/fig3.png',  dpi=600, bbox_inches='tight')

"""**DaDn Plots**"""

def create_da_dk_actual(filename, crack_length, change):
  file = pd.read_csv(filename)
  data_array = np.asarray(file)

  stress_intensity = data_array[:, 1].reshape(len(data_array), 1)
  stress_intensity = Scaler.transform(stress_intensity)

  da = []
  dk = []

  for i in range(len(crack_length) - change):
    da.append(crack_length[i + change] - crack_length[i])

  for i in range(len(stress_intensity) - change):
    dk.append(stress_intensity[i + change] - stress_intensity[i])

  return (np.asarray(da)), (np.asarray(dk))

def dadn_fn(x, c, n):
    return np.sign(x)*c*np.abs(x)**n

def create_da_dk_intermediate(crack_length, stress_intensity, change, use_inter):
  #use_inter = 1 #0=no 1=yes
  #
  # MATLAB CODE used for Experiment
  #
  #triang_x = [20 30];
  #triang_y = interp1(dK_I_1800(:,:,1), a_1800(:,:,1), triang_x);
  #slope = diff(triang_y)/diff(triang_x);
  #
  dadn = []
  dk = []

  minarr = int(np.min([len(crack_length),len(stress_intensity)]))
  startarr = int(0.5*minarr) #start at 50% of array
  stoparr = int(0.8*minarr) #stop at 80% of array
  if use_inter>0:
    dk = np.arange(20,80,1) #delta k
    a = np.asarray(crack_length[startarr:stoparr].flatten())
    k = np.asarray(stress_intensity[startarr:stoparr].flatten())
    len_arr = int(np.floor(len(a)/change))
    da_in = np.zeros(len_arr)
    dk_in = np.zeros(len_arr)
    for i in range(0,len_arr-1):
      da_in[i]=((a[(i+1)*change] - a[i*change])/(change/512*10))
      dk_in[i]=((k[(i+1)*change] - k[i*change])*np.sqrt(1000))

    print(da_in)
    print(dk_in)

    #interpolate function was 1d arrays...here we come reshape
    #dadn_f = interpolate.interp1d(dk_in,da_in,fill_value='extrapolate')
    #dadn = dadn_f(dk)
    popt, pcov = curve_fit(dadn_fn, dk_in, da_in,)
    print('fit: c=%10.3e, n=%10.3e' % tuple(popt))
    dadn = dadn_fn(dk, *popt)
    #slope = np.diff(triang_y)/np.diff(triang_x)
  else:
    #forward finite difference
    all_data = 0 #use fit data only=0 use all data =1
    if all_data > 0:
      len_arr = int(startarr/change)+int(np.floor((stoparr-startarr)/change))
      #crack length units coming in will have units mm and we want mm/cycle
      for i in range(int(startarr/change),len_arr-1):
        dadn.append((crack_length[(i+1)*change] - crack_length[i*change])/(change/512))

      #stress intensity coming in will have units MPa*sqrt(mm) and we want MPa*sqrt(m)
      for i in range(int(startarr/change),len_arr-1):
        dk.append((stress_intensity[(i+1)*change] - stress_intensity[i*change])*np.sqrt(1000))
    else:
      len_arr = int(np.floor(len(crack_length)/change))
      #crack length units coming in will have units mm and we want mm/cycle
      for i in range(0,len_arr-1):
        dadn.append((crack_length[(i+1)*change] - crack_length[i*change])/(change/512*10))

      #stress intensity coming in will have units MPa*sqrt(mm) and we want MPa*sqrt(m)
      for i in range(0,len_arr-1):
        dk.append((stress_intensity[(i+1)*change] - stress_intensity[i*change])*np.sqrt(1000))

  return np.asarray(dadn), np.asarray(dk)

change = 512
#python is driving me crazy with these 2d arrays with one column wtf python. that is why there are all these reshapes...
#interpolated da/dn values
print('1600 35C');da1600_35, dk1600_35 = create_da_dk_intermediate(yhat1600_35, calc_stress_intensity(force_1600_35,crack_1600_35,0).reshape((len(crack_1600_35),1)), change,1)
print('1700 35C');da1700_35, dk1700_35 = create_da_dk_intermediate(yhat1700_35, calc_stress_intensity(force_1700_35,crack_1700_35,0).reshape((len(crack_1700_35),1)), change,1)
print('1800 35C');da1800_35, dk1800_35 = create_da_dk_intermediate(yhat1800_35, calc_stress_intensity(force_1800_35,crack_1800_35,0).reshape((len(crack_1800_35),1)), change,1)

print('1600 75C');da1600_75, dk1600_75 = create_da_dk_intermediate(yhat1600_75, calc_stress_intensity(force_1600_75,crack_1600_75.reshape((len(crack_1600_75),1)),0), change,1)
print('1700 75C');da1700_75, dk1700_75 = create_da_dk_intermediate(yhat1700_75, calc_stress_intensity(force_1700_75,crack_1700_75.reshape((len(crack_1700_75),1)),0), change,1)
print('1800 75C');da1800_75, dk1800_75 = create_da_dk_intermediate(yhat1800_75, calc_stress_intensity(force_1800_75,crack_1800_75.reshape((len(crack_1800_75),1)),0), change,1)

print('1600 150C');da1600_150, dk1600_150 = create_da_dk_intermediate(yhat1600_150, calc_stress_intensity(force_1600_150,crack_1600_150.reshape((len(crack_1600_150),1)),0), change,1)
print('1700 150C');da1700_150, dk1700_150 = create_da_dk_intermediate(yhat1700_150, calc_stress_intensity(force_1700_150,crack_1700_150.reshape((len(crack_1700_150),1)),0), change,1)
print('1800 150C');da1800_150, dk1800_150 = create_da_dk_intermediate(yhat1800_150, calc_stress_intensity(force_1800_150,crack_1800_150.reshape((len(crack_1800_150),1)),0), change,1)

print('1600 200C');da1600_200, dk1600_200 = create_da_dk_intermediate(yhat1600_200, calc_stress_intensity(force_1600_200,crack_1600_200.reshape((len(crack_1600_200),1)),0), change,1)
print('1700 200C');da1700_200, dk1700_200 = create_da_dk_intermediate(yhat1700_200, calc_stress_intensity(force_1700_200,crack_1700_200.reshape((len(crack_1700_200),1)),0), change,1)
print('1800 200C');da1800_200, dk1800_200 = create_da_dk_intermediate(yhat1800_200, calc_stress_intensity(force_1800_200,crack_1800_200.reshape((len(crack_1800_200),1)),0), change,1)

print('1600 250C');da1600_250, dk1600_250 = create_da_dk_intermediate(yhat1600_250, calc_stress_intensity(force_1600_250,crack_1600_250.reshape((len(crack_1600_250),1)),0), change,1)
print('1700 250C');da1700_250, dk1700_250 = create_da_dk_intermediate(yhat1700_250, calc_stress_intensity(force_1700_250,crack_1700_250.reshape((len(crack_1700_250),1)),0), change,1)
print('1800 250C');da1800_250, dk1800_250 = create_da_dk_intermediate(yhat1800_250, calc_stress_intensity(force_1800_250,crack_1800_250.reshape((len(crack_1800_250),1)),0), change,1)

print('1600 300C');da1600_300, dk1600_300 = create_da_dk_intermediate(yhat1600_300, calc_stress_intensity(force_1600_300,crack_1600_300,0).reshape((len(crack_1600_300),1)), change,1)
print('1700 300C');da1700_300, dk1700_300 = create_da_dk_intermediate(yhat1700_300, calc_stress_intensity(force_1700_300,crack_1700_300,0).reshape((len(crack_1700_300),1)), change,1)
print('1800 300C');da1800_300, dk1800_300 = create_da_dk_intermediate(yhat1800_300, calc_stress_intensity(force_1800_300,crack_1800_300,0).reshape((len(crack_1800_300),1)), change,1)

print('1600 400C');da1600_400, dk1600_400 = create_da_dk_intermediate(yhat1600_400, calc_stress_intensity(force_1600_400,crack_1600_400,0).reshape((len(crack_1600_400),1)), change,1)
print('1700 400C');da1700_400, dk1700_400 = create_da_dk_intermediate(yhat1700_400, calc_stress_intensity(force_1700_400,crack_1700_400,0).reshape((len(crack_1700_400),1)), change,1)
print('1800 400C');da1800_400, dk1800_400 = create_da_dk_intermediate(yhat1800_400, calc_stress_intensity(force_1800_400,crack_1800_400,0).reshape((len(crack_1800_400),1)), change,1)

print('1600 500C');da1600_500, dk1600_500 = create_da_dk_intermediate(yhat1600_500, calc_stress_intensity(force_1600_500,crack_1600_500,0).reshape((len(crack_1600_500),1)), change,1)
print('1700 500C');da1700_500, dk1700_500 = create_da_dk_intermediate(yhat1700_500, calc_stress_intensity(force_1700_500,crack_1700_500,0).reshape((len(crack_1700_500),1)), change,1)
print('1800 500C');da1800_500, dk1800_500 = create_da_dk_intermediate(yhat1800_500, calc_stress_intensity(force_1800_500,crack_1800_500,0).reshape((len(crack_1800_500),1)), change,1)

#finite difference da/dn values
da1600_35_fd, dk1600_35_fd = create_da_dk_intermediate(yhat1600_35, calc_stress_intensity(force_1600_35,crack_1600_35,0).reshape((len(force_1600_35),1)), change,0)
da1700_35_fd, dk1700_35_fd = create_da_dk_intermediate(yhat1700_35, calc_stress_intensity(force_1700_35,crack_1700_35,0).reshape((len(force_1700_35),1)), change,0)
da1800_35_fd, dk1800_35_fd = create_da_dk_intermediate(yhat1800_35, calc_stress_intensity(force_1800_35,crack_1800_35,0).reshape((len(force_1800_35),1)), change,0)

da1600_75_fd, dk1600_75_fd = create_da_dk_intermediate(yhat1600_75, calc_stress_intensity(force_1600_75,crack_1600_75,0).reshape((len(force_1600_75),1)), change,0)
da1700_75_fd, dk1700_75_fd = create_da_dk_intermediate(yhat1700_75, calc_stress_intensity(force_1700_75,crack_1700_75,0).reshape((len(force_1700_75),1)), change,0)
da1800_75_fd, dk1800_75_fd = create_da_dk_intermediate(yhat1800_75, calc_stress_intensity(force_1800_75,crack_1800_75,0).reshape((len(force_1800_75),1)), change,0)

da1600_150_fd, dk1600_150_fd = create_da_dk_intermediate(yhat1600_150, calc_stress_intensity(force_1600_150,crack_1600_150.reshape((len(force_1600_150),1)),0), change,0)
da1700_150_fd, dk1700_150_fd = create_da_dk_intermediate(yhat1700_150, calc_stress_intensity(force_1700_150,crack_1700_150.reshape((len(force_1700_150),1)),0), change,0)
da1800_150_fd, dk1800_150_fd = create_da_dk_intermediate(yhat1800_150, calc_stress_intensity(force_1800_150,crack_1800_150.reshape((len(force_1800_150),1)),0), change,0)

da1600_200_fd, dk1600_200_fd = create_da_dk_intermediate(yhat1600_200, calc_stress_intensity(force_1600_200,crack_1600_200.reshape((len(force_1600_200),1)),0), change,0)
da1700_200_fd, dk1700_200_fd = create_da_dk_intermediate(yhat1700_200, calc_stress_intensity(force_1700_200,crack_1700_200.reshape((len(force_1700_200),1)),0), change,0)
da1800_200_fd, dk1800_200_fd = create_da_dk_intermediate(yhat1800_200, calc_stress_intensity(force_1800_200,crack_1800_200.reshape((len(force_1800_200),1)),0), change,0)

da1600_250_fd, dk1600_250_fd = create_da_dk_intermediate(yhat1600_250, calc_stress_intensity(force_1600_250,crack_1600_250.reshape((len(force_1600_250),1)),0), change,0)
da1700_250_fd, dk1700_250_fd = create_da_dk_intermediate(yhat1700_250, calc_stress_intensity(force_1700_250,crack_1700_250.reshape((len(force_1700_250),1)),0), change,0)
da1800_250_fd, dk1800_250_fd = create_da_dk_intermediate(yhat1800_250, calc_stress_intensity(force_1800_250,crack_1800_250.reshape((len(force_1800_250),1)),0), change,0)

da1600_300_fd, dk1600_300_fd = create_da_dk_intermediate(yhat1600_300, calc_stress_intensity(force_1600_300,crack_1600_300,0).reshape((len(force_1600_300),1)), change,0)
da1700_300_fd, dk1700_300_fd = create_da_dk_intermediate(yhat1700_300, calc_stress_intensity(force_1700_300,crack_1700_300,0).reshape((len(force_1700_300),1)), change,0)
da1800_300_fd, dk1800_300_fd = create_da_dk_intermediate(yhat1800_300, calc_stress_intensity(force_1800_300,crack_1800_300,0).reshape((len(force_1800_300),1)), change,0)

da1600_400_fd, dk1600_400_fd = create_da_dk_intermediate(yhat1600_400, calc_stress_intensity(force_1600_400,crack_1600_400,0).reshape((len(force_1600_400),1)), change,0)
da1700_400_fd, dk1700_400_fd = create_da_dk_intermediate(yhat1700_400, calc_stress_intensity(force_1700_400,crack_1700_400,0).reshape((len(force_1700_400),1)), change,0)
da1800_400_fd, dk1800_400_fd = create_da_dk_intermediate(yhat1800_400, calc_stress_intensity(force_1800_400,crack_1800_400,0).reshape((len(force_1800_400),1)), change,0)

da1600_500_fd, dk1600_500_fd = create_da_dk_intermediate(yhat1600_500, calc_stress_intensity(force_1600_500,crack_1600_500,0).reshape((len(force_1600_500),1)), change,0)
da1700_500_fd, dk1700_500_fd = create_da_dk_intermediate(yhat1700_500, calc_stress_intensity(force_1700_500,crack_1700_500,0).reshape((len(force_1700_500),1)), change,0)
da1800_500_fd, dk1800_500_fd = create_da_dk_intermediate(yhat1800_500, calc_stress_intensity(force_1800_500,crack_1800_500,0).reshape((len(force_1800_500),1)), change,0)

plt.figure(figsize=(15,15))
plt.rc('font', size=20)
#1600N
#plt.scatter(dk1600_35, da1600_35, label = '1600N @ 35C')
#plt.scatter(dk1600_75, da1600_75, label = '1600N @ 75C')
#plt.scatter(dk1600_150, da1600_150, label = '1600N @ 150C')
#plt.scatter(dk1600_250, da1600_250, label = '1600N @ 250C')
#plt.scatter(dk1600_300, da1600_300, label = '1600N @ 300C')
#plt.scatter(dk1600_300_fd, da1600_300_fd, label = '1600N @ 300C')
#plt.scatter(dk1600_400, da1600_400, label = '1600N @ 400C')
#plt.scatter(dk1600_500, da1600_500, label = '1600N @ 500C')

#1700N
#plt.scatter(dk1700_35_fd, da1700_35_fd, label = '1700N @ 35C')
plt.plot(dk1700_75, da1700_75, 'b^-', label = '1700N @ 75\u2103', linewidth=3, markersize=12,markevery=5)
#plt.scatter(dk1700_75, da1700_75, label = '1700N @ 75C')
#plt.scatter(dk1700_75_fd, da1700_75_fd, label = '1700N @ 75C')
#plt.scatter(dk1700_150, da1700_150, label = '1700N @ 150C')
plt.plot(dk1700_200, da1700_200, 'b>-', label = '1700N @ 200\u2103', linewidth=3, markersize=12,markevery=5)
#plt.scatter(dk1700_200, da1700_200, label = '1700N @ 200C')
#plt.scatter(dk1700_250, da1700_250, label = '1700N @ 250C')
plt.plot(dk1700_300, da1700_300, 'bx-', label = '1700N @ 300\u2103', linewidth=3, markersize=12,markevery=5)
#plt.plot(dk1700_300_fd, da1700_300_fd, 'b,-', label = '1700N @ 300C')
plt.plot(dk1700_400, da1700_400, 'bo-', label = '1700N @ 400\u2103', linewidth=3, markersize=12,markevery=5)
plt.plot(dk1700_500, da1700_500, 'bv-', label = '1700N @ 500\u2103', linewidth=3, markersize=12,markevery=5)

#1800N
#plt.scatter(dk1800_35_fd, da1800_35_fd, label = '1800N @ 35C')
plt.plot(dk1800_75, da1800_75, 'g^-', label = '1800N @ 75\u2103', linewidth=3, markersize=12,markevery=5)
#plt.scatter(dk1800_75_fd, da1800_75_fd, label = '1800N @ 75C')
#plt.scatter(dk1800_150, da1800_150, label = '180N @ 150C')
plt.plot(dk1800_200, da1800_200, 'g>-', label = '1800N @ 200\u2103', linewidth=3, markersize=12,markevery=5)
#plt.scatter(dk1800_250, da1800_250, label = '1800N @ 250C')
plt.plot(dk1800_300, da1800_300, 'gx-', label = '1800N @ 300\u2103', linewidth=3, markersize=12,markevery=5)
#plt.plot(dk1800_300_fd, da1800_300_fd, label = '1800N @ 300C')
plt.plot(dk1800_400, da1800_400, 'go-', label = '1800N @ 400\u2103', linewidth=3, markersize=12,markevery=5)
plt.plot(dk1800_500, da1800_500, 'gv-', label = '1800N @ 500\u2103', linewidth=3, markersize=12,markevery=5)


plt.xscale("log")
plt.yscale("log")
plt.xlim([10, 100])
plt.ylim([4e-6, 3e-3])
plt.xlabel('$\Delta$k (MPa-$\sqrt{m}$)')
plt.ylabel('$\Delta$a/$\Delta$n (mm/cycle)')
plt.legend()
plt.savefig('./Paper Images/fig4.pdf', format='pdf',bbox_inches='tight')