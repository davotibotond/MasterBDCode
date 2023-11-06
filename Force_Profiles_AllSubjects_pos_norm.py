#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 3 17:51:26 2023

@author: botonddavoti
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

### Input variables
sampling_frequency = 200  # Hz
data_directory = '/Users/botonddavoti/MasterPython/Data'
output_directory = "/Users/botonddavoti/MasterPython/Force Curves"
resistance_types = ['freeweight', 'keiser', 'quantum', 'norse']
dpi = 100
position_filter_threshold = 0.2  # 20 cm as meters

### Dictionary to map different terms to a unified terminology
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
    # Add more mappings if there are more terms
}

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to preprocess and plot data
def process_and_plot_data(df_dictionary, resistance_type, subject, exercise):
    for df_key_name, df_value_rep in df_dictionary.items():
        # Preprocess each dataframe/rep
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

        if Rep_con_start is None or Rep_con_end is None:
            print(f"Could not determine concentric phase for {df_key_name}. Skipping this rep.")
            continue  # Skip to the next dataframe/rep if the phase cannot be determined
        else:
            # Narrow down to the determined concentric phase
            df_value_rep = df_value_rep.iloc[Rep_con_start:Rep_con_end]

        # Replace dataframes in df_dictionary with new dataframes
        df_dictionary[df_key_name] = df_value_rep

    # Plot and save figures to PDF
    subject_output_directory = os.path.join(output_directory, subject)
    os.makedirs(subject_output_directory, exist_ok=True)
    pdf_path = os.path.join(subject_output_directory, f"{resistance_type}_{exercise}.pdf")
    with PdfPages(pdf_path) as pdf:
        for metric in ['Barbell position', 'Barbell velocity', 'Barbell force (FP)']:
            fig, ax = plt.subplots(dpi=dpi)
            for df_key_name, df_value_rep in df_dictionary.items():
                if metric == 'Barbell position':
                    # Filter by position threshold for Barbell position only
                    filtered_df = df_value_rep[df_value_rep[metric] > position_filter_threshold]
                    data_to_plot = filtered_df if not filtered_df.empty else df_value_rep
                    ax.plot(data_to_plot['timestamp'], data_to_plot[metric], label=df_key_name)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(f'{metric} (m)')  # Set y-axis label to meters
                else:
                    # No filtering for other metrics; plot against Barbell position
                    ax.plot(df_value_rep['timestamp'], df_value_rep[metric], label=df_key_name)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(f'{metric} ({'m/s' if metric == 'Barbell velocity' else 'N'})')  # Set y-axis label to m/s or N
                
                ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

# Loop through all subjects, exercises, and resistance types
for subject_folder in os.listdir(data_directory):
    subject_path = os.path.join(data_directory, subject_folder)
    if os.path.isdir(subject_path):
        for exercise_folder in os.listdir(subject_path):
            exercise_path = os.path.join(subject_path, exercise_folder)
            if os.path.isdir(exercise_path):
                for resistance_type in resistance_types:
                    resistance_path = os.path.join(exercise_path, resistance_type)
                    if os.path.isdir(resistance_path):
                        # Read all .csv files and store into df_dictionary
                        df_dictionary = {}
                        counter = 1
                        for file_name in os.listdir(resistance_path):
                            if file_name.endswith('.csv'):
                                file_path = os.path.join(resistance_path, file_name)
                                df = pd.read_csv(file_path, delimiter=';', decimal=",")
                                df.rename(columns=term_mapping, inplace=True)
                                df_key_name = f"Rep{counter}"
                                df_dictionary[df_key_name] = df
                                counter += 1

                        if df_dictionary:
                            # Process data and plot figures
                            process_and_plot_data(df_dictionary, resistance_type, subject_folder, exercise_folder)

print("Processing complete.")
