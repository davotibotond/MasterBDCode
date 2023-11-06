#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 3 16:25:23 2023

@author: botonddavoti
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

### Input variables
sampling_frequency = 200  # Hz

### Read all files in a folder (rep1-6) and put into one dataframe
folder_path = '/Users/botonddavoti/MasterPython/Data/Subject_1/squat/quantum'

df_dictionary = {}  # Dictionary to store the DataFrames
counter = 1  # Counter for DataFrame names

# Dictionary to map different terms to a unified terminology
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
    # Add more mappings if there are more terms
}

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.csv'):
        df_key_name = f"Rep{counter}"  # Construct DataFrame name
        df = pd.read_csv(file_path, delimiter=';', decimal=",")
        # Rename columns based on term_mapping
        df.rename(columns=term_mapping, inplace=True)
        df_dictionary[df_key_name] = df
        counter += 1  # Increment the counter

# Preprocess each dataframe/rep
for df_key_name, df_value_rep in df_dictionary.items():
    # Calculate force from force platforms
    df_value_rep['Barbell force (FP)'] = df_value_rep[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    
    df_value_rep.drop(
        ['Gulv stor sway X', 'Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X', 'Gulv v sway Y',
         'Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1, inplace=True)

    # Define start and end of concentric phase using barbell velocity
    Rep_velocity_cutoff = 0.015
    Number_samples_cutoff1 = round(0.4 * sampling_frequency)
    Number_samples_cutoff2 = round(0.0067 * sampling_frequency)
                
    Rep_con_start = None
    Rep_con_end = None
    
    for i in range(len(df_value_rep['timestamp']) - Number_samples_cutoff1 + 1):  
        if all(df_value_rep.loc[i:i + Number_samples_cutoff1 - 1, 'Barbell velocity'] > Rep_velocity_cutoff):
            Rep_con_start = i
            break
    
    if Rep_con_start is not None:  # Proceed only if start was found
        for i in range(Rep_con_start, len(df_value_rep['timestamp']) - Number_samples_cutoff2 + 1):  
            if all(df_value_rep.loc[i:i + Number_samples_cutoff2 - 1, 'Barbell velocity'] < Rep_velocity_cutoff):
                Rep_con_end = i + Number_samples_cutoff2 - 1
                break
    
    if Rep_con_start is not None and Rep_con_end is not None:
        df_value_rep = df_value_rep.iloc[Rep_con_start:Rep_con_end]
    else:
        print(f"Could not determine concentric phase for {df_key_name}. Skipping this rep.")
        continue  # Skip to the next dataframe/rep if the phase cannot be determined

    # Replace dataframes in df_dictionary with new dataframes
    df_dictionary[df_key_name] = df_value_rep

# Plot figures
dpi = 100
fig_position, ax_position = plt.subplots(dpi=dpi)
fig_velocity, ax_velocity = plt.subplots(dpi=dpi)
fig_force_FP, ax_force_FP = plt.subplots(dpi=dpi)

position_filter_threshold = 0.2  # 20 cm as meters

for df_key_name, df_value_rep in df_dictionary.items():
    # Filter the dataframe to include only the rows where 'Barbell position' is greater than the threshold
    filtered_df = df_value_rep[df_value_rep['Barbell position'] > position_filter_threshold]
    
    # Plot only if there are rows that meet the condition
    if not filtered_df.empty:
        # Plot barbell position
        ax_position.plot(filtered_df['Barbell position'], label=df_key_name)
        ax_position.set_xlabel('Barbell position (m)')
        ax_position.set_ylabel('Barbell position (m)')
        ax_position.legend()

        # Plot barbell velocity
        ax_velocity.plot(filtered_df['Barbell position'], filtered_df['Barbell velocity'], label=df_key_name)
        ax_velocity.set_xlabel('Barbell position (m)')
        ax_velocity.set_ylabel('Barbell velocity (m/s)')
        ax_velocity.legend()

        # Plot barbell force (FP)
        ax_force_FP.plot(filtered_df['Barbell position'], filtered_df['Barbell force (FP)'], label=df_key_name)
        ax_force_FP.set_xlabel('Barbell position (m)')
        ax_force_FP.set_ylabel('Barbell force (FP) (N)')
        ax_force_FP.legend()
    else:
        print(f"{df_key_name} does not have data above the position filter threshold and will not be plotted.")

plt.show()


# Ensure the output directory exists
output_directory = "/Users/botonddavoti/MasterPython/Force Curves"
os.makedirs(output_directory, exist_ok=True)

# Save the figures to the output directory
fig_position.savefig(os.path.join(output_directory, 'Barbell position.png'), dpi=dpi)
fig_velocity.savefig(os.path.join(output_directory, 'Barbell velocity.png'), dpi=dpi)
fig_force_FP.savefig(os.path.join(output_directory, 'Barbell force (FP).png'), dpi=dpi)

# Close the figures after saving
plt.close(fig_position)
plt.close(fig_velocity)
plt.close(fig_force_FP)

# End of script
