#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:30:30 2023

@author: botonddavoti
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.interpolate import interp1d

### Input variables
sampling_frequency = 200  # Hz
data_directory = './data'  # Update with the actual path to your data
output_directory = "./outputs/Power1"  # Update with your desired output path
resistance_types = ['freeweight', 'keiser', 'quantum', 'norse']
dpi = 100

### Color mapping for different resistance modalities
color_mapping = {
    'freeweight': 'blue',
    'keiser': 'orange',
    'quantum': 'grey',
    'norse': 'green'
}

### Dictionary to map different terms to a unified terminology
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
    # Add more mappings if there are more terms
}

# Butterworth Low-Pass Filter function
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Function to preprocess and plot data
def process_and_plot_data(df_dictionary, exercise):
    valid_reps = {}  # Dictionary to store only the valid reps

    for df_key_name, df_value_rep in df_dictionary.items():
        # Preprocess each dataframe/rep
        df_value_rep['Barbell force (FP)'] = df_value_rep[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
        df_value_rep['Power'] = df_value_rep['Barbell force (FP)'] * df_value_rep['Barbell velocity']

        # Apply Butterworth low-pass filter to the Power signal
        df_value_rep['Power'] = butter_lowpass_filter(df_value_rep['Power'], 20, sampling_frequency)

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

        if Rep_con_start is not None:
            for i in range(Rep_con_start, len(df_value_rep['timestamp']) - Number_samples_cutoff2 + 1):
                if all(df_value_rep.loc[i:i + Number_samples_cutoff2 - 1, 'Barbell velocity'] < Rep_velocity_cutoff):
                    Rep_con_end = i + Number_samples_cutoff2 - 1
                    break

        if Rep_con_start is None or Rep_con_end is None:
            print(f"Could not determine concentric phase for {df_key_name}. Skipping this rep.")
            continue

        df_con_phase = df_value_rep.iloc[Rep_con_start:Rep_con_end]

        # Normalize barbell position to a scale of 0-100%
        min_pos = df_con_phase['Barbell position'].min()
        max_pos = df_con_phase['Barbell position'].max()
        df_con_phase['Normalized position'] = (df_con_phase['Barbell position'] - min_pos) / (max_pos - min_pos) * 100

        # Store the processed rep
        valid_reps[df_key_name] = df_con_phase

    return valid_reps

# Function to interpolate data to a common set of normalized positions
def interpolate_data(data_list):
    # Filter out empty data sequences
    data_list = [data for data in data_list if not data.empty]

    if not data_list:
        return []

    min_normalized_pos = min(data['Normalized position'].min() for data in data_list)
    max_normalized_pos = max(data['Normalized position'].max() for data in data_list)
    common_normalized_pos = np.linspace(min_normalized_pos, max_normalized_pos, 100)  # Adjust the number of points as needed

    interpolated_data_list = []
    for data in data_list:
        f = interp1d(data['Normalized position'], data['Power'], kind='linear', fill_value='extrapolate')
        interpolated_power = f(common_normalized_pos)
        interpolated_data = pd.DataFrame({'Normalized position': common_normalized_pos, 'Power': interpolated_power})
        interpolated_data_list.append(interpolated_data)

    return interpolated_data_list

# Function to plot average power curve for each resistance type on the same graph
# Function to plot average power curve for each resistance type on the same graph
def plot_average_power_curves(exercise_data, output_directory, exercises):  

    for exercise in exercises:
        fig, ax = plt.subplots(dpi=dpi)
        for resistance_type in exercise_data[exercise].keys():
            aggregated_data = interpolate_data(exercise_data[exercise][resistance_type])
            if not aggregated_data:
                continue
            mean_power = np.mean([data['Power'] for data in aggregated_data], axis=0)
            # Use the color mapping and line style for the resistance type
            ax.plot(aggregated_data[0]['Normalized position'], mean_power, 
                    label=resistance_type,  
                    color=color_mapping[resistance_type])

        ax.set_xlabel('Barbell position (%)')
        ax.set_ylabel("Power (W)")
        ax.legend()
        ax.set_title(f"Power Curve - {exercise}")

        pdf_path = os.path.join(output_directory, f"{exercise}_average_power_curve.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        # Disable gridlines
        plt.grid(False)
        plt.close(fig)

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Updated exercises list
exercises = ['Bench', 'Squat', 'Row']  # Adjust these names to match your folder names

# Process data for each resistance type
exercise_data = {exercise: {resistance_type: [] for resistance_type in resistance_types} for exercise in exercises}

for subject_folder in os.listdir(data_directory):
    subject_path = os.path.join(data_directory, subject_folder)
    if os.path.isdir(subject_path):
        for exercise_folder in os.listdir(subject_path):
            if exercise_folder in exercises:
                exercise_path = os.path.join(subject_path, exercise_folder)
                if os.path.isdir(exercise_path):
                    for resistance_type in resistance_types:
                        resistance_path = os.path.join(exercise_path, resistance_type)
                        if os.path.isdir(resistance_path):
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
                                processed_data = process_and_plot_data(df_dictionary, exercise_folder)
                                exercise_data[exercise_folder][resistance_type].extend(processed_data.values())

# Plot and save PDFs with all resistance types on the same graph for each exercise
plot_average_power_curves(exercise_data, output_directory, exercises)

print("Processing complete.")
