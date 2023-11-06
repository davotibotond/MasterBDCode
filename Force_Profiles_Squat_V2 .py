#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:25:23 2023

@author: botonddavoti
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

### Input variables
sampling_frequency = 200  # Hz

### Read all files in a folder (rep1-6) and put into one dataframe
folder_path = r'/Users/botonddavoti/MasterPython/Data/Subject_1/squat/freeweight'

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
    # Calculate force from force platforms
    df_value_rep['Stang force (FP)'] = df_value_rep[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    
    df_value_rep = df_value_rep.drop(
        ['Gulv stor sway X', 'Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X', 'Gulv v sway Y',
         'Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1)

    # Define start and end of con phase using barbell velocity
    Rep_velocity_cutoff = 0.015
    Number_samples_cutoff1 = round(0.4 * sampling_frequency)
    Number_samples_cutoff2 = round(0.0067 * sampling_frequency)
                
    Rep_con_start = None
    Rep_con_end = None
    
    for i in range(len(df_value_rep['timestamp']) - Number_samples_cutoff1 + 1):  
        if all(df_value_rep.loc[i:i + Number_samples_cutoff1 - 1, 'Stang velocity'] > Rep_velocity_cutoff):
            Rep_con_start = i
            break
    
    for i in range(Rep_con_start, len(df_value_rep['timestamp']) - Number_samples_cutoff2 + 1):  
        if all(df_value_rep.loc[i:i + Number_samples_cutoff2 - 1, 'Stang velocity'] < Rep_velocity_cutoff):
            Rep_con_end = i + Number_samples_cutoff2 - 1
            break
    
    df_value_rep = df_value_rep.iloc[Rep_con_start:Rep_con_end]

    # Replace dataframes in df_dictionary with new dataframes
    df_dictionary[df_key_name] = df_value_rep

# Plot figures
dpi = 300
fig_position, ax_position = plt.subplots(dpi=dpi)
fig_velocity, ax_velocity = plt.subplots(dpi=dpi)
fig_force_FP, ax_force_FP = plt.subplots(dpi=dpi)

for df_key_name, df_value_rep in df_dictionary.items():
    # Plot barbell position
    ax_position.plot(df_value_rep['timestamp'], df_value_rep['Stang position'], label=df_key_name)
    ax_position.set_xlabel('Time (ms)')
    ax_position.set_ylabel('Stang position (m)')
    ax_position.legend()

    # Plot barbell velocity
    ax_velocity.plot(df_value_rep['Stang position'], df_value_rep['Stang velocity'], label=df_key_name)
    ax_velocity.set_xlabel('Position (m)')
    ax_velocity.set_ylabel('Stang velocity (m/s)')
    ax_velocity.legend()

    # Plot barbell force (FP)
    ax_force_FP.plot(df_value_rep['Stang position'], df_value_rep['Stang force (FP)'], label=df_key_name)
    ax_force_FP.set_xlabel('Position (m)')
    ax_force_FP.set_ylabel('Stang force (FP) (N)')
    ax_force_FP.legend()

# Display the figures interactively
plt.show()

# Output directory
output_directory = "/Users/botonddavoti/MasterPython/Force Curves"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Save the figures to the output directory
fig_position.savefig(os.path.join(output_directory, 'Stang position.png'), dpi=dpi)
fig_velocity.savefig(os.path.join(output_directory, 'Stang velocity.png'), dpi=dpi)
fig_force_FP.savefig(os.path.join(output_directory, 'Stang force (FP).png'), dpi=dpi)

# Close the figures after saving
plt.close(fig_position)
plt.close(fig_velocity)
plt.close(fig_force_FP)
