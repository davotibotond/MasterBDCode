#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:24:45 2023

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

### Sørg for at utdatamappen eksisterer
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

### Ordbok for å mappe forskjellige termer til en enhetlig terminologi
term_mapping = {
    'Stang position': 'Barbell position',
    'Stang velocity': 'Barbell velocity',
    'Stang force (FP)': 'Barbell force (FP)',
}

# Funksjon for å forbehandle og velge gyldige data
def preprocess_and_select_data(df, term_mapping, sampling_frequency, exercise_name):
    df.rename(columns=term_mapping, inplace=True)

    if exercise_name == 'squat':
        df['Barbell force (FP)'] = df[['Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    else:
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
        print(f"Kunne ikke bestemme konsentrisk fase for repetisjon i filen. Hopper over dette repetisjonen.")
        return None  # Ugyldig repetisjon, hopp over denne

    df_con_phase = df.iloc[Rep_con_start:Rep_con_end]

    if df_con_phase['Barbell position'].max() <= 0.25:
        print(f"Maksimal posisjon er ikke større enn 0,25 m. Hopper over dette repetisjonen.")
        return None  # Hopp over dette repetisjonen hvis maksimal posisjon ikke er større enn 0,25 m

    return df_con_phase

# Initsialiser datastrukturen for å lagre alle forbehandlede data
all_data = {}

# Hovedløkke med spesifikk behandling for forskjellige øvelser
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

# Funksjon for å beregne gjennomsnittskurven
def calculate_average_curve(resistance_data, all_positions):
    all_forces = []

    for df_key_name, df_value_rep in resistance_data.items():
        interp_func = interp1d(df_value_rep['Barbell position'], df_value_rep['Barbell force (FP)'],
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_forces = interp_func(all_positions)
        interpolated_forces[interpolated_forces < 0] = 0
        all_forces.append(interpolated_forces)

    avg_force = np.mean(all_forces, axis=0)
    return avg_force

# Funksjon for å generere og lagre plottene med separate akser
def generate_plots_with_dual_axes(all_data, output_directory, dpi):
    common_positions = np.linspace(0, 100, num=1000)  # Felles sett med posisjoner for interpolasjon
    for resistance_type in resistance_types:
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=dpi)
        ax2 = None  # Initialiser en annen y-akse

        for exercise_name, exercise_data in all_data.items():
            if resistance_type in exercise_data:
                resistance_data = exercise_data[resistance_type]
                avg_force = calculate_average_curve(resistance_data, common_positions)
                if exercise_name == 'squat':
                    if ax2 is None:
                        ax2 = ax1.twinx()  # Opprett en annen y-akse for "squat"
                    ax2.plot(common_positions, avg_force, label=exercise_name, linestyle='--')
                else:
                    ax1.plot(common_positions, avg_force, label=exercise_name)

        ax1.set_xlabel('Stangposisjon (%)')
        ax1.set_ylabel('Kraft (FP) Bench/Row')
        if ax2 is not None:
            ax2.set_ylabel('Kraft (FP) Squat')
        ax1.set_title(f'Gjennomsnittlige kraftkurver - {resistance_type}')
        ax1.legend(loc='upper left')
        if ax2 is not None:
            ax2.legend(loc='upper right')
        ax1.grid(True)
        plt.tight_layout()

        # Lagre plottet som en PDF
        pdf_filename = f"{resistance_type}_average_force_curves.pdf"
        pdf_path = os.path.join(output_directory, pdf_filename)
        plt.savefig(pdf_path)
        plt.close()

# Bruk den nye funksjonen for å generere plottene
generate_plots_with_dual_axes(all_data, output_directory, dpi)

print("Behandling fullført. Sjekk utdatamappen for plottene med separate akser og titler.")
