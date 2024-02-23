#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:16:39 2023

@author: botonddavoti
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import spm1d
import itertools

### Input variables
sampling_frequency = 200  # Hz
data_directory = './data'  # Update with the actual path to your data
output_directory = "./outputs/SPM2"  # Update with your desired output path
resistance_types = ['freeweight', 'keiser', 'quantum', 'norse']
dpi = 100

# Define color mapping for resistance types
color_mapping = {
    'freeweight': 'blue',
    'keiser': 'orange',
    'quantum': 'grey',
    'norse': 'green'
}

### Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

### Dictionary to map different terms to a unified terminology
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
}

# Function to preprocess and select valid data with specific handling for different exercises
def preprocess_and_select_data(df, term_mapping, sampling_frequency, exercise_name):
    df.rename(columns=term_mapping, inplace=True)

    # Specific handling for "squat" exercise
    if exercise_name in ['squat', 'bench', 'row']:
        df['Barbell force (FP)'] = df[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    else:  # For other exercises
        # Handle other exercises if needed
        pass
    
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
        print(f"Could not determine concentric phase for rep in file. Skipping this rep.")
        return None  # Invalid rep, skip this one

    df_con_phase = df.iloc[Rep_con_start:Rep_con_end]

    if df_con_phase['Barbell position'].max() <= 0.25:
        print(f"Max position is not greater than 0.25m. Skipping this rep.")
        return None  # Skip this rep if max position is not greater than 0.25m

    return df_con_phase

# Initialize the data structure to store all preprocessed data
all_data = {}

# Main processing loop with specific handling for different exercises
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
                                df_con_phase = preprocess_and_select_data(df, term_mapping, sampling_frequency, exercise_name)
                                if df_con_phase is not None:
                                    df_key_name = f"{resistance_type}_{file_name}"
                                    df_dictionary[df_key_name] = df_con_phase
                        if df_dictionary:
                            exercise_data[resistance_type] = df_dictionary
                if exercise_data:
                    all_data[exercise_name] = exercise_data


# Function to identify sticking regions in the exercises squat and bench press
def mark_lowest_position_in_range(mean_curve, positions, start_pct, end_pct):
    start_idx = int(start_pct / 100 * len(positions))
    end_idx = int(end_pct / 100 * len(positions))
    segment = mean_curve[start_idx:end_idx]
    min_idx = np.argmin(segment) + start_idx
    min_value = mean_curve[min_idx]
    min_position = positions[min_idx]
    return min_value, min_position

# Function to generate and save combined SPM analysis and mean force curve plots
def generate_combined_pdf(all_data, output_directory, dpi, resistance_types):
    common_positions = np.linspace(0, 100, num=100)  # Define common positions for interpolation

    for exercise_name, exercise_data in all_data.items():
        pdf_filename = f"{exercise_name}_combined_analysis.pdf"
        pdf_path = os.path.join(output_directory, pdf_filename)
        
        with PdfPages(pdf_path) as pdf:
            # Set up the pairs for each exercise
            if exercise_name == 'squat':
                pairs_sets = [[('freeweight', 'keiser'), ('freeweight', 'quantum'), ('keiser', 'quantum')]]
            else:  # 'bench' and 'row'
                pairs_sets = [
                    [('freeweight', 'keiser'), ('freeweight', 'norse'), ('freeweight', 'quantum')],
                    [('keiser', 'norse'), ('keiser', 'quantum'), ('quantum', 'norse')]
                ]

            for pairs in pairs_sets:
                fig, axs = plt.subplots(2, 3, figsize=(11, 8.5), gridspec_kw={'height_ratios': [1.0, 0.5]}, dpi=dpi)
                fig.suptitle(f"{exercise_name.capitalize()} Analysis - Page {pairs_sets.index(pairs) + 1}")

                # Iterate over each pair and generate plots
                for i, pair in enumerate(pairs):
                    if pair[0] in exercise_data and pair[1] in exercise_data:
                        # Extract and interpolate data for the SPM analysis
                        data1 = [df['Barbell force (FP)'].values for df in exercise_data[pair[0]].values()]
                        data2 = [df['Barbell force (FP)'].values for df in exercise_data[pair[1]].values()]
                        data1_interp = [np.interp(common_positions, np.linspace(0, 100, len(d)), d) for d in data1]
                        data2_interp = [np.interp(common_positions, np.linspace(0, 100, len(d)), d) for d in data2]
                        data1_mean = np.mean(data1_interp, axis=0)
                        data2_mean = np.mean(data2_interp, axis=0)
                        data1_std = np.std(data1_interp, axis=0)
                        data2_std = np.std(data2_interp, axis=0)
                        data1 = np.array(data1_interp)
                        data2 = np.array(data2_interp)

                        # Perform SPM t-test
                        t = spm1d.stats.ttest2(data1, data2, equal_var=True).inference(alpha=0.05)

                        # Plot SPM t-test results
                        ax_spm = axs[1, i]
                        ax_spm.set_title(f'SPM {pair[0]} vs {pair[1]}')
                        t.plot(ax=ax_spm)
                        t.plot_threshold_label(ax=ax_spm)
                        # This is inside the `generate_combined_pdf` function, within the loop iterating over pairs
                        if any(p < 0.05 for p in t.p):  # Check if any p-value is significant
                            t.plot_p_values(ax=ax_spm)

                        # Plot mean force curves 
                        ax_mfc = axs[0, i]
                        ax_mfc.set_title(f'Force {pair[0]} vs {pair[1]}')
                        ax_mfc.plot(common_positions, data1_mean, label=f'{pair[0]} Mean', color=color_mapping[pair[0]])
                        ax_mfc.fill_between(common_positions, data1_mean - data1_std, data1_mean + data1_std, alpha=0.2, color=color_mapping[pair[0]])
                        ax_mfc.plot(common_positions, data2_mean, label=f'{pair[1]} Mean', color=color_mapping[pair[1]])
                        ax_mfc.fill_between(common_positions, data2_mean - data2_std, data2_mean + data2_std, alpha=0.2, color=color_mapping[pair[1]])

                        if exercise_name in ['squat', 'bench']:  # Add this check
                            # Mark the lowest position in the 20-60% range for both curves
                            min_value1, min_position1 = mark_lowest_position_in_range(data1_mean, common_positions, 20, 60)
                            min_value2, min_position2 = mark_lowest_position_in_range(data2_mean, common_positions, 20, 60)
                            ax_mfc.scatter(min_position1, min_value1, color='black')  # Mark on first curve
                            ax_mfc.scatter(min_position2, min_value2, color='black')  # Mark on second curve

                        # We have removed the annotations for standard deviation
                        # Set y-axis limits based on the exercise name
                        if exercise_name == 'squat':
                            ax_mfc.set_ylim(750, 2900)
                        elif exercise_name == 'bench':
                            ax_mfc.set_ylim(150, 750)
                        elif exercise_name == 'row':
                            ax_mfc.set_ylim(100, 900)

                        ax_mfc.legend()

                        # Adding labels for x-axis and y-axis
                        ax_mfc.set_xlabel('Barbell Position (%)')
                        ax_mfc.set_ylabel('Force (N)')

                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter as needed
                pdf.savefig(fig)
                plt.close(fig)

# Example usage of the function
generate_combined_pdf(all_data, output_directory, dpi, resistance_types)

print("Combined PDF generation complete. Check the output directory for the PDFs.")


