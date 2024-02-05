#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:49:36 2023

@author: botonddavoti
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages

### Input variables
sampling_frequency = 200  # Hz
data_directory = './data'  # Update with the actual path to your data
output_directory = "./outputs/test"  # Update with your desired output path
resistance_types = ['freeweight', 'keiser', 'quantum', 'norse']
dpi = 100

### Dictionary to map different terms to a unified terminology
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
    # Add more mappings if there are more terms
}

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to preprocess and select valid data
def preprocess_and_select_data(df, term_mapping, sampling_frequency):
    # Apply term mapping to rename columns
    df.rename(columns=term_mapping, inplace=True)
    df['Barbell force (FP)'] = df[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    df.drop(['Gulv stor sway X', 'Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X', 'Gulv v sway Y',
             'Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1, inplace=True)

    # Normalize barbell position to percentage
    max_position = df['Barbell position'].max()
    df['Barbell position'] = (df['Barbell position'] / max_position) * 100  # Convert to percentage

    # Define start and end of concentric phase using barbell velocity
    Rep_velocity_cutoff = 0.015
    Number_samples_cutoff1 = round(0.4 * sampling_frequency)
    Number_samples_cutoff2 = round(0.0067 * sampling_frequency)

    Rep_con_start = None
    Rep_con_end = None

    for i in range(len(df['timestamp']) - Number_samples_cutoff1 + 1):
        if all(df.loc[i:i + Number_samples_cutoff1 - 1, 'Barbell velocity'] > Rep_velocity_cutoff):
            Rep_con_start = i
            break

    if Rep_con_start is not None:
        for i in range(Rep_con_start, len(df['timestamp']) - Number_samples_cutoff2 + 1):
            if all(df.loc[i:i + Number_samples_cutoff2 - 1, 'Barbell velocity'] < Rep_velocity_cutoff):
                Rep_con_end = i + Number_samples_cutoff2 - 1
                break

    if Rep_con_start is None or Rep_con_end is None:
        print(f"Could not determine concentric phase for rep in file. Skipping this rep.")
        return None  # Invalid rep, skip this one

    df_con_phase = df.iloc[Rep_con_start:Rep_con_end]

    if df_con_phase['Barbell position'].max() <= 0.25:
        print(f"Max position is not greater than 0.25m. Skipping this rep.")
        return None  # Skip this rep if max position is not greater than 0.25m

    return df_con_phase

# Main processing loop
all_data = {}
for subject_folder in os.listdir(data_directory):
    subject_path = os.path.join(data_directory, subject_folder)
    if os.path.isdir(subject_path):
        for exercise_folder in os.listdir(subject_path):
            exercise_path = os.path.join(subject_path, exercise_folder)
            if os.path.isdir(exercise_path):
                exercise_name = exercise_folder.lower()
                exercise_data = {}
                for resistance_type in resistance_types:
                    resistance_path = os.path.join(exercise_path, resistance_type)
                    if os.path.isdir(resistance_path):
                        df_dictionary = {}
                        counter = 1
                        for file_name in os.listdir(resistance_path):
                            if file_name.endswith('.csv'):
                                file_path = os.path.join(resistance_path, file_name)
                                df = pd.read_csv(file_path, delimiter=';', decimal=",")
                                df_con_phase = preprocess_and_select_data(df, term_mapping, sampling_frequency)
                                if df_con_phase is not None:
                                    df_key_name = f"Rep{counter}"
                                    df_dictionary[df_key_name] = df_con_phase
                                    counter += 1

                        if df_dictionary:
                            exercise_data[resistance_type] = df_dictionary

                if exercise_data:
                    all_data[exercise_name] = exercise_data

# Plot and save figures for all exercises and resistance types
for exercise_name, exercise_data in all_data.items():
    for resistance_type, resistance_data in exercise_data.items():
        pdf_path = os.path.join(output_directory, f"{exercise_name}_{resistance_type}.pdf")
        with PdfPages(pdf_path) as pdf:
            for metric in ['Barbell force (FP)', 'Barbell velocity']:
                fig, ax = plt.subplots(dpi=dpi)
                for df_key_name, df_value_rep in resistance_data.items():
                    ax.plot(df_value_rep['Barbell position'], df_value_rep[metric], label=df_key_name)
                ax.set_xlabel('Barbell position (%)')
                ax.set_ylabel(metric)
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)

print("Processing complete.")
