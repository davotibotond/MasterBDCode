# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:26:24 2023

@author: lassem
"""

### Delete console and variables
#clear #cls # or clear
#%reset # or locals().clear() or globals().clear()



### Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

 

### Input variables
sampling_frequency = 200 #Hz



### Read all files in a folder (rep1-6) and put into one dataframe
folder_path = r'/Users/botonddavoti/MasterPython/Data/Subject_15/Row/Keiser'


df_dictionary = {}  # Dictionary to store the DataFrames; data structures that store key-value pairs
counter = 1  # Counter for DataFrame names

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.csv'):
        df_key_name = f"Rep{counter}"  # Construct DataFrame name
        df_dictionary[df_key_name] = pd.read_csv(file_path, delimiter=';', decimal=",")
        counter += 1  # Increment the counter



# Preprocess each dataframe/rep
for df_key_name, df_value_rep in df_dictionary.items(): #df_key_name could be any other name (first key, then value)
    
    # Delete unnecessary columns
    df_value_rep = df_value_rep.drop(
        ['Gulv stor sway X','Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X','Gulv v sway Y'], axis=1)
        
    # Define start and end of con phase using barbell velocity (and delete unnecessary rows)
            # Could also consider to use barbell position peak, especially for Rep_con_end (or a combination of barbell position and velocity)
    Rep_velocity_cutoff = 0.015
    Number_samples_cutoff1 = round(0.4 * sampling_frequency)  # i.e. 100 samples (0.5 sec); For start of ecc and con phase
    Number_samples_cutoff2 = round(0.0067 * sampling_frequency)  # i.e. 1 sample (0.0067 sec); For end of ecc and con phase
                
    Rep_con_start = None
    Rep_con_end = None
    
    for i in range(len(df_value_rep['timestamp'])):
        if all(df_value_rep.loc[range(i,i + Number_samples_cutoff1),'Barbell velocity'] > Rep_velocity_cutoff):
            Rep_con_start = i
            break
    
    for i in range(len(df_value_rep['timestamp'])):
        if all(df_value_rep.loc[range(i,i + Number_samples_cutoff2),'Barbell velocity'] < Rep_velocity_cutoff) and df_value_rep.index[i] > Rep_con_start:
            Rep_con_end = i
            break
    
    df_value_rep = df_value_rep.iloc[Rep_con_start:Rep_con_end]
                  
    
    # Calculate force from force platforms
    df_value_rep['Barbell force (FP)'] = df_value_rep[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    
    df_value_rep = df_value_rep.drop(
        ['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1)

    # Reset timestamp to 0
    df_value_rep['timestamp'] = df_value_rep['timestamp'].sub(df_value_rep['timestamp'].min())
        #Alt: df_value_rep['timestamp'] = df_value_rep['timestamp']-df_value_rep.loc[127,'timestamp'] # Requires reset of index
    
    # Replace dataframes in df_dictionary with new dataframes
    df_dictionary[df_key_name] = df_value_rep 



### Plot figures
dpi = 300
fig_position, ax_position = plt.subplots(dpi=dpi)
fig_velocity, ax_velocity = plt.subplots(dpi=dpi)
fig_power, ax_power = plt.subplots(dpi=dpi)
fig_force, ax_force = plt.subplots(dpi=dpi)
fig_force_FP, ax_force_FP = plt.subplots(dpi=dpi)



for df_key_name, df_value_rep in df_dictionary.items():  

    # Plot barbell position
    ax_position.plot(df_value_rep['timestamp'], df_value_rep['Barbell position'], label=df_key_name)
    ax_position.set_xlabel('Time (ms)')
    ax_position.set_ylabel('Barbell position (m)')
    #ax_position.set_title('Barbell position')
    ax_position.legend()

        
    # Plot barbell velocity
    ax_velocity.plot(df_value_rep['timestamp'], df_value_rep['Barbell velocity'], label=df_key_name)
    ax_velocity.set_xlabel('Time (ms)')
    ax_velocity.set_ylabel('Barbell velocity (m/s)')
    #ax_velocity.set_title('Barbell velocity')
    ax_velocity.legend()

 
    # Plot barbell power
    ax_power.plot(df_value_rep['timestamp'], df_value_rep['Barbell power'], label=df_key_name)
    ax_power.set_xlabel('Time (ms)')
    ax_power.set_ylabel('Barbell power (W)')
    #ax_power.set_title('Barbell power')
    ax_power.legend()
    
    
    # Plot barbell force
    ax_force.plot(df_value_rep['timestamp'], df_value_rep['Barbell force'], label=df_key_name)
    ax_force.set_xlabel('Time (ms)')
    ax_force.set_ylabel('Barbell force (N)')
    #ax_force.set_title('Barbell force')
    ax_force.legend()


    # Plot barbell force (FP)
    ax_force_FP.plot(df_value_rep['timestamp'], df_value_rep['Barbell force (FP)'], label=df_key_name)
    ax_force_FP.set_xlabel('Time (ms)')
    ax_force_FP.set_ylabel('Barbell force (FP) (N)')
    #ax_force_FP.set_title('Barbell force (FP)')
    ax_force_FP.legend()



fig_position.savefig(os.path.join(folder_path, '1. Barbell position.png'), dpi=dpi)
fig_velocity.savefig(os.path.join(folder_path, '2. Barbell velocity.png'), dpi=dpi)
fig_power.savefig(os.path.join(folder_path, '3. Barbell power.png'), dpi=dpi)
fig_force.savefig(os.path.join(folder_path, '4. Barbell force.png'), dpi=dpi)
fig_force_FP.savefig(os.path.join(folder_path, '5. Barbell force (FP).png'), dpi=dpi)




### Helping code
#df_bench_freeweight = pd.read_csv(r'C:\D\Extra files\PhD\Benkpress\Frivekt + Muscle Lab\Bench Press FP_Mausehund Lasse_31-05-2023_16-12-49.csv', delimiter=';', decimal=",")
#print(df_bench_freeweight['Barbell velocity'])
#print(df_bench_freeweight.loc[0,'Barbell velocity'])
#type(df_bench_freeweight.loc[0,'Barbell velocity'])

