#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:24:33 2023

@author: botonddavoti
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import tab20

### Input variables
sampling_frequency = 200  # Hz
data_directory = './data'  # Update with the actual path to your data
output_directory = "./outputs/test1"  # Update with your desired output path
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

    max_position = df['Barbell position'].max()
    df['Barbell position'] = (df['Barbell position'] / max_position) * 100

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
        return None

    df_con_phase = df.iloc[Rep_con_start:Rep_con_end]

    if df_con_phase['Barbell position'].max() <= 0.25:
        return None

    return df_con_phase

# Function to interpolate forces
def interpolate_forces(valid_positions, valid_forces, all_positions):
    min_valid_position = min(valid_positions)
    max_valid_position = max(valid_positions)
    positions_to_interpolate = all_positions[(all_positions >= min_valid_position) & (all_positions <= max_valid_position)]
    interp_method = 'linear' if len(valid_positions) < 4 else 'cubic'
    interp_func = interp1d(valid_positions, valid_forces, kind=interp_method, fill_value='extrapolate')
    interpolated_forces = interp_func(positions_to_interpolate)
    full_interpolated_forces = np.zeros_like(all_positions)
    full_interpolated_forces[(all_positions >= min_valid_position) & (all_positions <= max_valid_position)] = interpolated_forces
    full_interpolated_forces[all_positions < min_valid_position] = valid_forces.iloc[0]
    full_interpolated_forces[all_positions > max_valid_position] = valid_forces.iloc[-1]
    return full_interpolated_forces

# Function to create a list of distinguishable colors
def get_distinct_colors(n):
    color_norm = plt.Normalize(vmin=0, vmax=n-1)
    scalar_map = plt.cm.ScalarMappable(norm=color_norm, cmap=tab20)
    return [scalar_map.to_rgba(i) for i in range(n)]

# Modified function to process and plot data for each subject
def calculate_and_plot_subject_curves(all_data, output_directory, dpi):
    for exercise_name, exercise_data in all_data.items():
        for resistance_type, resistance_data in exercise_data.items():
            fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
            colors = get_distinct_colors(len(resistance_data))

            sorted_subjects = sorted(resistance_data.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
            all_avg_forces = []
            all_peak_forces = []

            for idx, subject in enumerate(sorted_subjects):
                subject_data = resistance_data[subject]
                subject_forces = []
                all_positions = np.linspace(0, 100, 1000)

                for df in subject_data:
                    valid_positions = df['Barbell position']
                    valid_forces = df['Barbell force (FP)']
                    interpolated_forces = interpolate_forces(valid_positions, valid_forces, all_positions)
                    subject_forces.append(interpolated_forces)

                avg_force = np.mean(subject_forces, axis=0)
                all_avg_forces.append(avg_force)
                all_peak_forces.append(max(avg_force))
                ax.plot(all_positions, avg_force, label=f'{subject}', color=colors[idx])  # <-- Change made here

            # Calculate the spread and standard deviation
            avg_force_spread = np.ptp([np.mean(force) for force in all_avg_forces])
            peak_force_spread = max(all_peak_forces) - min(all_peak_forces)
            std_dev_avg_force = np.std([np.mean(force) for force in all_avg_forces])
            std_dev_peak_force = np.std(all_peak_forces)

            # Plotting the calculated values on the graph
            ax.text(1.05, 0.5, f'Avg Force Spread: {avg_force_spread:.2f}\n'
                                f'Peak Force Spread: {peak_force_spread:.2f}\n'
                                f'Std Dev Avg Force: {std_dev_avg_force:.2f}\n'
                                f'Std Dev Peak Force: {std_dev_peak_force:.2f}',
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes)

            ax.set_title(f'Average Force Curve for {exercise_name} - {resistance_type}')
            ax.set_xlabel('Barbell Position (%)')
            ax.set_ylabel('Force (FP)')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=5)
            ax.grid(True)
            plt.tight_layout()

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
        subject_id = subject_folder
        for exercise_folder in os.listdir(subject_path):
            exercise_path = os.path.join(subject_path, exercise_folder)
            if os.path.isdir(exercise_path):
                exercise_name = exercise_folder.lower()
                if exercise_name not in all_data:
                    all_data[exercise_name] = {}
                for resistance_type in resistance_types:
                    resistance_path = os.path.join(exercise_path, resistance_type)
                    if os.path.isdir(resistance_path):
                        subject_resistance_data = []
                        for file_name in os.listdir(resistance_path):
                            if file_name.endswith('.csv'):
                                file_path = os.path.join(resistance_path, file_name)
                                df = pd.read_csv(file_path, delimiter=';', decimal=",")
                                df_con_phase = preprocess_and_select_data(df, term_mapping, sampling_frequency)
                                if df_con_phase is not None:
                                    subject_resistance_data.append(df_con_phase)
                        if subject_resistance_data:
                            if resistance_type not in all_data[exercise_name]:
                                all_data[exercise_name][resistance_type] = {}
                            all_data[exercise_name][resistance_type][subject_id] = subject_resistance_data

calculate_and_plot_subject_curves(all_data, output_directory, dpi)

print("Processing complete. Check the output directory for the plots.")
