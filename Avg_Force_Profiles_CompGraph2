import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages

### Input variables
sampling_frequency = 200  # Hz
data_directory = '/Users/botonddavoti/MasterPython/Data 2'  # Update with the actual path to your data
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

# Modified function to preprocess and select valid data with specific handling for different exercises
def preprocess_and_select_data(df, term_mapping, sampling_frequency, exercise_name):
    df.rename(columns=term_mapping, inplace=True)

    # Specific handling for "squat" exercise
    if exercise_name == 'squat':
        df['Barbell force (FP)'] = df[['Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
    else:  # For "bench", "row", and other exercises
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

# Function to calculate the average curve across all reps for each resistance type
def calculate_average_curve(resistance_data, all_positions):
    # Prepare data for averaging
    all_forces = []

    # Collect force data from all reps
    for df_key_name, df_value_rep in resistance_data.items():
        # Interpolate force data
        interp_func = interp1d(df_value_rep['Barbell position'], df_value_rep['Barbell force (FP)'],
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_forces = interp_func(all_positions)
        # Replace negative values with zero
        interpolated_forces[interpolated_forces < 0] = 0
        all_forces.append(interpolated_forces)

    # Calculate the average force across all reps
    avg_force = np.mean(all_forces, axis=0)
    return avg_force

# Generate and save the plots
def generate_plots(all_data, output_directory, dpi):
    common_positions = np.linspace(0, 100, num=1000)  # Common set of positions for interpolation
    for exercise_name, exercise_data in all_data.items():
        plt.figure(figsize=(10, 5), dpi=dpi)
        for resistance_type, resistance_data in exercise_data.items():
            avg_force = calculate_average_curve(resistance_data, common_positions)
            plt.plot(common_positions, avg_force, label=f'{resistance_type} Average')
        plt.title(f'Average Force Curves - {exercise_name}')
        plt.xlabel('Barbell Position (%)')
        plt.ylabel('Force (FP)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot to a PDF
        pdf_filename = f"{exercise_name}_average_force_curves.pdf"
        pdf_path = os.path.join(output_directory, pdf_filename)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig()
        plt.close()

generate_plots(all_data, output_directory, dpi)

print("Processing complete. Check the output directory for the plots.")
