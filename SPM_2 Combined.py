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
output_directory = "./outputs/SPM3"  # Update with your desired output path
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

# Function to calculate freeweight average force
def calculate_freeweight_avg_force(exercise_data):
    freeweight_forces = []

    if 'freeweight' in exercise_data:
        freeweight_data = exercise_data['freeweight']
        for df_key_name, df_value_rep in freeweight_data.items():
            freeweight_forces.extend(df_value_rep['Barbell force (FP)'].values)

    if freeweight_forces:
        return np.mean(freeweight_forces)
    else:
        return None

# Function to calculate the average curve
def calculate_average_curve_and_std(resistance_data, all_positions):
    all_forces = []

    for df_key_name, df_value_rep in resistance_data.items():
        interp_func = interp1d(df_value_rep['Barbell position'], df_value_rep['Barbell force (FP)'],
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_forces = interp_func(all_positions)
        interpolated_forces[interpolated_forces < 0] = 0
        all_forces.append(interpolated_forces)

    avg_force = np.mean(all_forces, axis=0)
    std_deviation = np.std(all_forces, axis=0)  # Calculate standard deviation

    return avg_force, std_deviation

# Updated function to generate and save plots with standard deviation
def generate_plots_with_std(all_data, output_directory, dpi):
    common_positions = np.linspace(0, 100, num=1000)
    for exercise_name, exercise_data in all_data.items():
        plt.figure(figsize=(10, 5), dpi=dpi)
        for resistance_type, resistance_data in exercise_data.items():
            avg_force, std_deviation = calculate_average_curve_and_std(resistance_data, common_positions)
            plt.plot(common_positions, avg_force, label=f'{resistance_type} Mean')
            plt.fill_between(common_positions, avg_force - std_deviation, avg_force + std_deviation, alpha=0.2, label=f'{resistance_type} Std Dev')
        plt.title(f'Mean Force Curves with Standard Deviation - {exercise_name}')
        plt.xlabel('Barbell Position (%)')
        plt.ylabel('Force (FP)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        pdf_filename = f"{exercise_name}_mean_force_curves_with_std.pdf"
        pdf_path = os.path.join(output_directory, pdf_filename)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig()
        plt.close()

# Call the modified function to generate plots with standard deviation
generate_plots_with_std(all_data, output_directory, dpi)

print("Processing complete. Check the output directory for the plots.")

# SPM Analysis Function
def spm_analysis_with_std_and_numbers(all_data, output_directory, dpi):
    for exercise_name, exercise_data in all_data.items():
        pairs = itertools.combinations(resistance_types, 2)
        for pair in pairs:
            if pair[0] in exercise_data and pair[1] in exercise_data:
                data1 = [df['Barbell force (FP)'].values for df in exercise_data[pair[0]].values()]
                data2 = [df['Barbell force (FP)'].values for df in exercise_data[pair[1]].values()]

                # Interpolate data to have the same length
                common_positions = np.linspace(0, 100, num=100)
                data1_interp = [np.interp(common_positions, np.linspace(0, 100, len(d)), d) for d in data1]
                data2_interp = [np.interp(common_positions, np.linspace(0, 100, len(d)), d) for d in data2]

                # Calculate mean and standard deviation for each resistance type
                data1_mean = np.mean(data1_interp, axis=0)
                data1_std = np.std(data1_interp, axis=0)
                data2_mean = np.mean(data2_interp, axis=0)
                data2_std = np.std(data2_interp, axis=0)

                # Convert to numpy arrays for SPM analysis
                data1 = np.array(data1_interp)
                data2 = np.array(data2_interp)

                # Perform SPM t-test
                t = spm1d.stats.ttest2(data1, data2, equal_var=True).inference(alpha=0.05)

                # Create PDF with SPM results and mean force curves with standard deviation
                with PdfPages(os.path.join(output_directory, f'SPM_{exercise_name}_{pair[0]}_{pair[1]}.pdf')) as pdf:
                    # Plot SPM t-test results
                    plt.figure(figsize=(10, 7), dpi=dpi)
                    plt.subplot(2, 1, 1)
                    plt.title(f'SPM t Test {exercise_name} - {pair[0]} vs {pair[1]}')
                    t.plot()
                    t.plot_threshold_label()
                    if any(p < 0.05 for p in t.p):  # Check if any p-value is significant
                        t.plot_p_values()

                    # Plot mean force curves with standard deviation and add numerical annotations
                    plt.subplot(2, 1, 2)
                    plt.title(f'Mean Force Curves with Standard Deviation - {exercise_name}')
                    plt.plot(common_positions, data1_mean, label=f'{pair[0]} Mean')
                    plt.fill_between(common_positions, data1_mean - data1_std, data1_mean + data1_std, alpha=0.2)
                    plt.plot(common_positions, data2_mean, label=f'{pair[1]} Mean')
                    plt.fill_between(common_positions, data2_mean - data2_std, data2_mean + data2_std, alpha=0.2)
                    
                    # Choose positions to annotate standard deviation
                    positions_to_annotate = [25, 50, 75]  # Example positions in percentage
                    for pos in positions_to_annotate:
                        idx = int(pos / 100 * len(common_positions))
                        plt.annotate(f'±{data1_std[idx]:.2f}', 
                                     xy=(common_positions[idx], data1_mean[idx]), 
                                     textcoords="offset points", 
                                     xytext=(0,10), 
                                     ha='center')
                        plt.annotate(f'±{data2_std[idx]:.2f}', 
                                     xy=(common_positions[idx], data2_mean[idx]), 
                                     textcoords="offset points", 
                                     xytext=(0,-15), 
                                     ha='center')

                    plt.xlabel('Normalized Position (%)')
                    plt.ylabel('Force (FP)')
                    plt.legend()

                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            else:
                print(f"Data for pair {pair} not available in {exercise_name}, skipping this comparison.")

# Call the SPM analysis function
spm_analysis_with_std_and_numbers(all_data, output_directory, dpi)

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
                fig, axs = plt.subplots(2, 3, figsize=(11, 8.5), dpi=dpi)  # 2 rows and 3 columns
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
                        ax_spm = axs[0, i]
                        ax_spm.set_title(f'SPM {pair[0]} vs {pair[1]}')
                        t.plot(ax=ax_spm)
                        t.plot_threshold_label(ax=ax_spm)
                        if any(p < 0.05 for p in t.p):
                            # Plot p-values with numerical annotations for better visibility
                            for j, p in enumerate(t.p):
                                if p < 0.05:
                                    ax_spm.text(j, t.zstar + 0.02, f'p={p:.2f}', ha='center')

                        # Plot mean force curves with standard deviation and add numerical annotations
                        ax_mfc = axs[1, i]
                        ax_mfc.set_title(f'Mean Force {pair[0]} vs {pair[1]}')
                        ax_mfc.plot(common_positions, data1_mean, label=f'{pair[0]} Mean')
                        ax_mfc.fill_between(common_positions, data1_mean - data1_std, data1_mean + data1_std, alpha=0.2)
                        ax_mfc.plot(common_positions, data2_mean, label=f'{pair[1]} Mean')
                        ax_mfc.fill_between(common_positions, data2_mean - data2_std, data2_mean + data2_std, alpha=0.2)
                        
                        # Choose positions to annotate standard deviation
                        positions_to_annotate = [25, 50, 75]  # Example positions in percentage
                        for pos in positions_to_annotate:
                            idx = int(pos / 100 * len(common_positions))
                            ax_mfc.annotate(f'±{data1_std[idx]:.2f}', 
                                            xy=(common_positions[idx], data1_mean[idx]), 
                                            textcoords="offset points", 
                                            xytext=(0,10), 
                                            ha='center')
                            ax_mfc.annotate(f'±{data2_std[idx]:.2f}', 
                                            xy=(common_positions[idx], data2_mean[idx]), 
                                            textcoords="offset points", 
                                            xytext=(0,-15), 
                                            ha='center')

                        ax_mfc.legend()

                # Adjust layout and save the figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter as needed
                pdf.savefig(fig)
                plt.close(fig)

# Example usage of the function
generate_combined_pdf(all_data, output_directory, dpi, resistance_types)

print("Combined PDF generation complete. Check the output directory for the PDFs.")


