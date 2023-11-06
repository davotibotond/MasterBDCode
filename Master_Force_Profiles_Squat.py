# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:26:24 2023
@author: lassem
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### Input variables
sampling_frequency = 200 #Hz

### Read all files in a folder (rep1-6) and put into one dataframe
folder_path = r'/Users/botonddavoti/MasterPython/Data/Subject_15/squat/keiser'

df_dictionary = {}  # Dictionary to store the DataFrames
counter = 1  # Counter for DataFrame names

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.csv'):
        df_key_name = f"Rep{counter}"  # Construct DataFrame name
        df_dictionary[df_key_name] = pd.read_csv(file_path, delimiter=';', decimal=",")
        counter += 1  # Increment the counter

# Preprocess each dataframe/rep
for df_key_name, df_value_rep in df_dictionary.items():
    # Delete unnecessary columns
    df_value_rep = df_value_rep.drop(
        ['Gulv stor sway X','Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X','Gulv v sway Y'], axis=1)

    # Define start and end of con phase using barbell velocity (and delete unnecessary rows)
    Rep_velocity_cutoff = 0.015
    Number_samples_cutoff1 = round(0.4 * sampling_frequency)  # i.e. 100 samples (0.5 sec); For start of ecc and con phase
    Number_samples_cutoff2 = round(0.0067 * sampling_frequency)  # i.e. 1 sample (0.0067 sec); For end of ecc and con phase
                
    Rep_con_start = None
    Rep_con_end = None
    
    for i in range(len(df_value_rep['timestamp'])):
        if all(df_value_rep.loc[range(i,i + Number_samples_cutoff1),'Stang velocity'] > Rep_velocity_cutoff):
            Rep_con_start = i
            break
    
    for i in range(len(df_value_rep['timestamp'])):
        if all(df_value_rep.loc[range(i,i + Number_samples_cutoff2),'Stang velocity'] < Rep_velocity_cutoff) and df_value_rep.index[i] > Rep_con_start:
            Rep_con_end = i
            break
    
    df_value_rep = df_value_rep.iloc[Rep_con_start:Rep_con_end]
                  
    # Calculate force from force platforms
    df_value_rep['Stang force (FP)'] = df_value_rep[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    
    df_value_rep = df_value_rep.drop(
        ['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1)

    # Reset timestamp to 0
    df_value_rep['timestamp'] = df_value_rep['timestamp'].sub(df_value_rep['timestamp'].min())
    
    # Replace dataframes in df_dictionary with new dataframes
    df_dictionary[df_key_name] = df_value_rep 

# Output directory
output_directory = "/Users/botonddavoti/MasterPython/Force Curves"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Plot figures
dpi = 300
fig_position, ax_position = plt.subplots(dpi=dpi)
fig_velocity, ax_velocity = plt.subplots(dpi=dpi)
fig_power, ax_power = plt.subplots(dpi=dpi)
fig_force, ax_force = plt.subplots(dpi=dpi)
fig_force_FP, ax_force_FP = plt.subplots(dpi=dpi)

for df_key_name, df_value_rep in df_dictionary.items():
    # Plot barbell position
    ax_position.plot(df_value_rep['timestamp'], df_value_rep['Stang position'], label=df_key_name)
    ax_position.set_xlabel('Time (ms)')
    ax_position.set_ylabel('Stang position (m)')
    ax_position.legend()

    # Plot barbell velocity
    ax_velocity.plot(df_value_rep['timestamp'], df_value_rep['Stang velocity'], label=df_key_name)
    ax_velocity.set_xlabel('Time (ms)')
    ax_velocity.set_ylabel('Stang velocity (m/s)')
    ax_velocity.legend()

    # Plot barbell power
    ax_power.plot(df_value_rep['timestamp'], df_value_rep['Stang power'], label=df_key_name)
    ax_power.set_xlabel('Time (ms)')
    ax_power.set_ylabel('Stang power (W)')
    ax_power.legend()
    
    # Plot barbell force
    ax_force.plot(df_value_rep['timestamp'], df_value_rep['Stang force'], label=df_key_name)
    ax_force.set_xlabel('Time (ms)')
    ax_force.set_ylabel('Stang force (N)')
    ax_force.legend()

    # Plot barbell force (FP)
    ax_force_FP.plot(df_value_rep['timestamp'], df_value_rep['Stang force (FP)'], label=df_key_name)
    ax_force_FP.set_xlabel('Time (ms)')
    ax_force_FP.set_ylabel('Stang force (FP) (N)')
    ax_force_FP.legend()

fig_position.savefig(os.path.join(output_directory, '1. Stang position.png'), dpi=dpi)
fig_velocity.savefig(os.path.join(output_directory, '2. Stang velocity.png'), dpi=dpi)
fig_power.savefig(os.path.join(output_directory, '3. Stang power.png'), dpi=dpi)
fig_force.savefig(os.path.join(output_directory, '4. Stang force.png'), dpi=dpi)
fig_force_FP.savefig(os.path.join(output_directory, '5. Stang force (FP).png'), dpi=dpi)
