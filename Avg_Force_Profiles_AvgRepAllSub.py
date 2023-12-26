#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:22:27 2023

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
data_directory = '/Users/botonddavoti/MasterPython/Data'  # Update with the actual path to your data
output_directory = "/Users/botonddavoti/MasterPython/Average Force Curves"  # Update with your desired output path
resistance_types = ['freeweight', 'keiser', 'quantum', 'norse']
dpi = 100

### Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

### Dictionary to map different terms to a unified terminology
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
}

# Function to preprocess and select valid data
def preprocess_and_select_data(df, term_mapping, sampling_frequency):
    df.rename(columns=term_mapping, inplace=True)
    df['Barbell force (FP)'] = df[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    df.drop(['Gulv stor sway X', 'Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X', 'Gulv v sway Y',
             'Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1, inplace=True)

    # Normalize barbell position to percentage
    max_position = df['Barbell position'].max()
    df['Barbell position'] = (df['Barbell position'] / max_position) * 100

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
        return None  # Invalid rep, skip this one

    df_con_phase = df.iloc[Rep_con_start:Rep_con_end]

    if df_con_phase['Barbell position'].max() <= 0.25:
        return None  # Skip this rep if max position is not greater than 0.25m

    return df_con_phase

# Function to calculate the average curve across all reps for each exercise and resistance type
def calculate_and_plot_average_curves(all_data, output_directory, dpi):
    for exercise_name, exercise_data in all_data.items():
        for resistance_type, resistance_data in exercise_data.items():
            # Prepare data for averaging
            all_forces = []
            all_positions = np.linspace(0, 100, 1000)  # Common set of positions for interpolation

            # Collect force data from all reps
            for df_key_name, df_value_rep in resistance_data.items():
                # Interpolate force data
                valid_positions = df_value_rep['Barbell position']
                valid_forces = df_value_rep['Barbell force (FP)']
                interp_func = interp1d(valid_positions, valid_forces, kind='linear', bounds_error=False, fill_value='extrapolate')
                interpolated_forces = interp_func(all_positions)
                interpolated_forces[interpolated_forces < 0] = 0  # Correct negative values
                all_forces.append(interpolated_forces)

            # Calculate the average force across all reps
            avg_force = np.mean(all_forces, axis=0)

            # Plot the average force curve
            plt.figure(figsize=(10, 5), dpi=dpi)
            plt.plot(all_positions, avg_force, label='Average Force')
            plt.title(f'Average Force Curve for {exercise_name} - {resistance_type}')
            plt.xlabel('Barbell Position (%)')
            plt.ylabel('Force (FP)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot to a PDF
            pdf_filename = f"{exercise_name}_{resistance_type}_average_force_curve.pdf"
            pdf_path = os.path.join(output_directory, pdf_filename)
            with PdfPages(pdf_path) as pdf:
                pdf.savefig()
            plt.close()

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
                        for file_name in os.listdir(resistance_path):
                            if file_name.endswith('.csv'):
                                file_path = os.path.join(resistance_path, file_name)
                                df = pd.read_csv(file_path, delimiter=';', decimal=",")
                                df_con_phase = preprocess_and_select_data(df, term_mapping, sampling_frequency)
                                if df_con_phase is not None:
                                    df_key_name = f"{resistance_type}_{file_name}"
                                    df_dictionary[df_key_name] = df_con_phase
                        if df_dictionary:
                            exercise_data[resistance_type] = df_dictionary
                if exercise_data:
                    all_data[exercise_name] = exercise_data

# Calculate and plot the average curve for each exercise and resistance type
calculate_and_plot_average_curves(all_data, output_directory, dpi)

print("Processing complete. Check the output directory for the plots.")
