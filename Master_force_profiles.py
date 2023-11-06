# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:26:24 2023

@author: lassem
"""

### Delete console and variables
#clear #cls # or clear
#%reset # or locals().clear() or globals().clear()



### Import libraries ###
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import copy





### Input variables ###
sampling_frequency = 200 #Hz

                                  
      
                    
      
### Read all files and put into one dataframe ###
main_folder = r'C:\D\Extra files\PhD\Undervisning THP201 - 2022\Veiledning\Botond'
subject_folders = [f'Subject_{i}' for i in range(1, 2)]  # Specify the subject folders
    #[f'Subject_{i}' for i in range(1, 16)] 
exercise_folders = ['bench'] #['bench', 'row', 'squat']  # Specify the exercise folders
equipment_folders = ['freeweight'] #['freeweight', 'keiser', 'norse', 'quantum']  # Specify the equipment folders

# Create a dictionary structure to store the data in in DataFrames; data structures that store key-value pairs
    #Alt: MultiIndex DataFrame or File-based Storage (like BP approach)
all_data = {}

# Loop through the subject folders
for subject_folder in subject_folders:
    subject_path = os.path.join(main_folder, subject_folder)
    
    # Create a dictionary to store exercise data for the current subject
    exercise_data = {}
    
    # Loop through the exercise folders within each subject folder
    for exercise_folder in exercise_folders:
        exercise_path = os.path.join(subject_path, exercise_folder)
        
        # Create a dictionary to store equipment data for the current exercise
        equipment_data = {}

        
        # Loop through the equipment folders within each exercise folder
        for equipment_folder in equipment_folders:
            equipment_path = os.path.join(exercise_path, equipment_folder)
            
            counter = 1  # Counter for DataFrame names
            
            # Create a list to store the dataframes for the current equipment
            set_data = {}
            
            # Loop through the CSV files within each equipment folder
            for file_name in os.listdir(equipment_path):
                file_path = os.path.join(equipment_path, file_name)
                
                
                # Read all files in the equipment folder (rep1-6) and put into one dataframe dictionary
                if os.path.isfile(file_path) and file_name.endswith('.csv'):
                    df_key_name = f"Rep{counter}"  # Construct DataFrame name
                    set_data[df_key_name] = pd.read_csv(file_path, delimiter=';', decimal=",")
                    set_data[df_key_name]['File path'] = file_path
                    counter += 1  # Increment the counter
                    
                    # Also possible to process each DataFrame here                    

            
            # Store the list of dataframes for the current equipment in the equipment dictionary
            equipment_data[equipment_folder] = set_data
        
        # Store the equipment dictionary for the current exercise in the exercise dictionary
        exercise_data[exercise_folder] = equipment_data
    
    # Store the exercise dictionary for the current subject in the data structure
    all_data[subject_folder] = exercise_data





### Preprocess each dataframe/rep ###
# Loop through all subject/exercise/equipment/set data frames and the corresponding df_value_rep
for subject_folder, exercise_data in all_data.items():
    for exercise_folder, equipment_data in exercise_data.items():
        for equipment_folder, set_data in equipment_data.items():
            for df_key_rep_name, df_value_rep in set_data.items(): #df_key_rep_name could be any other name (first key, then value)
               
                try:             
                    # Delete unnecessary columns
                    df_value_rep = df_value_rep.drop(
                        ['Gulv stor sway X','Gulv stor sway Y', 'Gulv h sway X', 'Gulv h sway Y', 'Gulv v sway X','Gulv v sway Y'], axis=1)
                        
                    # Define start and end of con phase using barbell velocity (and delete unnecessary rows)
                            # Could also consider to use barbell position peak, especially for Rep_con_end (or a combination of barbell position and velocity)
                    Rep_velocity_cutoff = 0.015
                    Number_samples_cutoff1 = round(0.4 * sampling_frequency)  # i.e. 100 samples (0.5 sec); For start of ecc and con phase
                    Number_samples_cutoff2 = round(0.0067 * sampling_frequency)  # i.e. 1 sample (0.0067 sec); For end of ecc and con phase
                                
                    Rep_con_start = None
                    Rep_con_end = None
                    
                    for i in range(len(df_value_rep['timestamp'])):
                        if all(df_value_rep.loc[range(i,i + Number_samples_cutoff1),'Barbell velocity'] > Rep_velocity_cutoff):
                            Rep_con_start = i
                            break
                    
                    for i in range(len(df_value_rep['timestamp'])):
                        if all(df_value_rep.loc[range(i,i + Number_samples_cutoff2),'Barbell velocity'] < Rep_velocity_cutoff) and df_value_rep.index[i] > Rep_con_start: #or barbell position at peak
                            Rep_con_end = i
                            break
                    
                    df_value_rep = df_value_rep.iloc[Rep_con_start:Rep_con_end]
                                  
                    
                    # Calculate force from force platforms
                    df_value_rep['Barbell force (FP)'] = df_value_rep[['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton']].sum(axis=1)
                    
                    df_value_rep = df_value_rep.drop(
                        ['Gulv stor Newton', 'Gulv h Newton', 'Gulv v Newton'], axis=1)
                
                    # Reset timestamp to 0 and index to 0
                    df_value_rep['timestamp'] = df_value_rep['timestamp'].sub(df_value_rep['timestamp'].min())
                        #Alt: df_value_rep['timestamp'] = df_value_rep['timestamp']-df_value_rep.loc[127,'timestamp'] # Requires reset of index
                    
                    df_value_rep = df_value_rep.reset_index(drop=True)
                    
                    # Replace dataframes in set_data with new dataframes
                    set_data[df_key_rep_name] = df_value_rep 


                except Exception as e:
                    print('Error with the following file:')
                    print(set_data[df_key_rep_name].loc[0,'File path'])
                    print(f"The error is: {e}")

                    continue




### Time normalize each dataframe/rep ###
all_data_time_normalized = copy.deepcopy(all_data)


# Loop through all subject/exercise/equipment/set data frames and the corresponding df_value_rep
for subject_folder, exercise_data in all_data_time_normalized.items():
    for exercise_folder, equipment_data in exercise_data.items():
        for equipment_folder, set_data in equipment_data.items():
            for df_key_rep_name, df_value_rep in set_data.items(): 
               
                
                x = np.linspace(0, 99, len(df_value_rep['timestamp']))
                new_x = np.arange(100)
                
                # Create a new DataFrame with the desired index length
                df_normalized = pd.DataFrame(index=new_x)
                
                # Loop through each column of df_value_rep; # Time normalization 0-100% x-axis
                for column in df_value_rep.columns:
                    if column != 'timestamp' and column != 'File path':
                        spline_interpolator = interp1d(x, df_value_rep[column], kind='cubic')
                        time_normalized_values = spline_interpolator(new_x)
                        df_normalized[column] = time_normalized_values
                        #df_value_reps = time_normalized_values
                        #df_value_rep[column] = time_normalized_values
                
                
                # Assign the normalized values back to the original DataFrame
                df_normalized['File path'] = df_value_rep['File path'].iloc[0:100]
                set_data[df_key_rep_name] = df_normalized                                 
                    
                
                #Matlab codes for time normalization:
                    #x = linspace(0, 99, length(VARIABLE))
                    #y = 0:99;
                    ## Time normalization 0-100% x-axis
                    #TimeNormalizedVARIABLE = spline(x, VARIABLE, y);
                    ## Timenormalization 0-100% x-axis
    
                    # for i=1:Number_reps
                    #     Normalized_samples_con(:,i) = linspace(Repetition_events(k+2), Repetition_events(k+3), 4900); 
                    #     Elbow_angle_interpol_con(:,i) = interp1(Frame_3D(Repetition_events(k+2):Repetition_events(k+3)),Elbow_angle(Repetition_events(k+2):Repetition_events(k+3)), Normalized_samples_con(:,i),'linear')';
                    #     k=k+4;
                    # end
            
   

### Calculate the average of (all) ... reps
for subject_folder, exercise_data in all_data_time_normalized.items():
    for exercise_folder, equipment_data in exercise_data.items():
        for equipment_folder, set_data in equipment_data.items():
            
            # Create an empty dictionary to store the average values
            columns_to_average = {}
            
            # Iterate over the rows/reps
            for row in set_data.keys():         
                # Iterate over the columns
                for column in set_data[row].columns:
                    # Exclude the 'File path' column
                    if column != 'File path':
                        # Check if the column already exists in the columns_to_average dictionary
                        #columns_to_average[column] = pd.concat([columns_to_average[column], set_data[row][column]], axis=1)                      
                        
                        if column in columns_to_average:
                            # Concatenate the corresponding columns horizontally and calculate the mean
                            columns_to_average[column] = pd.concat([columns_to_average[column], set_data[row][column]], axis=1)
                        else:
                            # If the column does not exist in the columns_to_average dictionary, add it
                            columns_to_average[column] = set_data[row][column]

            averaged_columns = {}          
            for key, value in columns_to_average.items():
                averaged_columns[key] = value.mean(axis=1)    
            

            # Create a new dataframe using the columns_to_average dictionary
            set_data['Average_all_reps'] = pd.DataFrame(averaged_columns)

            if 'Rep1' in set_data: 
                set_data['Average_all_reps']['File path'] = set_data['Rep1']['File path']


            # Barbell_force_average = pd.concat(
            #     [set_data['Rep1']['Barbell force'], set_data['Rep2']['Barbell force'], set_data['Rep3']['Barbell force']
            #      , set_data['Rep4']['Barbell force'], set_data['Rep5']['Barbell force'], set_data['Rep6']['Barbell force']
            #      ], axis=1).mean(axis=1)





### Plot figures (all reps, without time normalization) ###

for subject_folder, exercise_data in all_data.items():
    for exercise_folder, equipment_data in exercise_data.items():
        for equipment_folder, set_data in equipment_data.items():
 
            dpi = 300
            fig_position, ax_position = plt.subplots(dpi=dpi)
            fig_velocity, ax_velocity = plt.subplots(dpi=dpi)
            fig_power, ax_power = plt.subplots(dpi=dpi)
            fig_force, ax_force = plt.subplots(dpi=dpi)
            fig_force_FP, ax_force_FP = plt.subplots(dpi=dpi)    

    
            for df_key_rep_name, df_value_rep in set_data.items():

                # Plot barbell position
                ax_position.plot(df_value_rep['timestamp'], df_value_rep['Barbell position'], label=df_key_rep_name)
                ax_position.set_xlabel('Time (ms)')
                ax_position.set_ylabel('Barbell position (m)')
                #ax_position.set_title('Barbell position')
                ax_position.legend()
            
                    
                # Plot barbell velocity
                ax_velocity.plot(df_value_rep['timestamp'], df_value_rep['Barbell velocity'], label=df_key_rep_name)
                ax_velocity.set_xlabel('Time (ms)')
                ax_velocity.set_ylabel('Barbell velocity (m/s)')
                #ax_velocity.set_title('Barbell velocity')
                ax_velocity.legend()
            
             
                # Plot barbell power
                ax_power.plot(df_value_rep['timestamp'], df_value_rep['Barbell power'], label=df_key_rep_name)
                ax_power.set_xlabel('Time (ms)')
                ax_power.set_ylabel('Barbell power (W)')
                #ax_power.set_title('Barbell power')
                ax_power.legend()
                
                
                # Plot barbell force
                ax_force.plot(df_value_rep['timestamp'], df_value_rep['Barbell force'], label=df_key_rep_name)
                ax_force.set_xlabel('Time (ms)')
                ax_force.set_ylabel('Barbell force (N)')
                #ax_force.set_title('Barbell force')
                ax_force.legend()
            
            
                # Plot barbell force (FP)
                ax_force_FP.plot(df_value_rep['timestamp'], df_value_rep['Barbell force (FP)'], label=df_key_rep_name)
                ax_force_FP.set_xlabel('Time (ms)')
                ax_force_FP.set_ylabel('Barbell force (FP) (N)')
                #ax_force_FP.set_title('Barbell force (FP)')
                ax_force_FP.legend()
    
 
                # Save all figures
                
                save_folder = os.path.join(os.path.dirname(set_data[df_key_rep_name].iloc[0]['File path']), 'Figures (all reps)')
                os.makedirs(save_folder, exist_ok=True) #Create the folder if it doesn't exist
                
                # Save the figure in the created folder
                fig_position.savefig(os.path.join(save_folder, '1. Barbell position.png'), dpi=dpi)
                fig_velocity.savefig(os.path.join(save_folder, '2. Barbell velocity.png'), dpi=dpi)
                fig_power.savefig(os.path.join(save_folder, '3. Barbell power.png'), dpi=dpi)
                fig_force.savefig(os.path.join(save_folder, '4. Barbell force.png'), dpi=dpi)
                fig_force_FP.savefig(os.path.join(save_folder, '5. Barbell force (FP).png'), dpi=dpi)                
                


            # Close the figures
            plt.close(fig_position)
            plt.close(fig_velocity)
            plt.close(fig_power)
            plt.close(fig_force)
            plt.close(fig_force_FP)







### Plot figures (all reps, with time normalization) ###

for subject_folder, exercise_data in all_data_time_normalized.items():
    for exercise_folder, equipment_data in exercise_data.items():
        for equipment_folder, set_data in equipment_data.items():
 
            dpi = 300
            fig_position, ax_position = plt.subplots(dpi=dpi)
            fig_velocity, ax_velocity = plt.subplots(dpi=dpi)
            fig_power, ax_power = plt.subplots(dpi=dpi)
            fig_force, ax_force = plt.subplots(dpi=dpi)
            fig_force_FP, ax_force_FP = plt.subplots(dpi=dpi)    

    
            for df_key_rep_name, df_value_rep in set_data.items():

                # Plot barbell position
                ax_position.plot(df_value_rep['Barbell position'], label=df_key_rep_name)
                ax_position.set_xlabel('Time (%)')
                ax_position.set_ylabel('Barbell position (m)')
                #ax_position.set_title('Barbell position')
                ax_position.legend()
            
                    
                # Plot barbell velocity
                ax_velocity.plot(df_value_rep['Barbell velocity'], label=df_key_rep_name)
                ax_velocity.set_xlabel('Time (%)')
                ax_velocity.set_ylabel('Barbell velocity (m/s)')
                #ax_velocity.set_title('Barbell velocity')
                ax_velocity.legend()
            
             
                # Plot barbell power
                ax_power.plot(df_value_rep['Barbell power'], label=df_key_rep_name)
                ax_power.set_xlabel('Time (%)')
                ax_power.set_ylabel('Barbell power (W)')
                #ax_power.set_title('Barbell power')
                ax_power.legend()
                
                
                # Plot barbell force
                ax_force.plot(df_value_rep['Barbell force'], label=df_key_rep_name)
                ax_force.set_xlabel('Time (%)')
                ax_force.set_ylabel('Barbell force (N)')
                #ax_force.set_title('Barbell force')
                ax_force.legend()
            
            
                # Plot barbell force (FP)
                ax_force_FP.plot(df_value_rep['Barbell force (FP)'], label=df_key_rep_name)
                ax_force_FP.set_xlabel('Time (%)')
                ax_force_FP.set_ylabel('Barbell force (FP) (N)')
                #ax_force_FP.set_title('Barbell force (FP)')
                ax_force_FP.legend()
    
 
                # Save all figures
                
                save_folder = os.path.join(os.path.dirname(set_data[df_key_rep_name].iloc[0]['File path']), 'Figures (all reps, time normalized)')
                os.makedirs(save_folder, exist_ok=True) #Create the folder if it doesn't exist
                
                # Save the figure in the created folder
                fig_position.savefig(os.path.join(save_folder, '1. Barbell position.png'), dpi=dpi)
                fig_velocity.savefig(os.path.join(save_folder, '2. Barbell velocity.png'), dpi=dpi)
                fig_power.savefig(os.path.join(save_folder, '3. Barbell power.png'), dpi=dpi)
                fig_force.savefig(os.path.join(save_folder, '4. Barbell force.png'), dpi=dpi)
                fig_force_FP.savefig(os.path.join(save_folder, '5. Barbell force (FP).png'), dpi=dpi)                
                


            # Close the figures
            plt.close(fig_position)
            plt.close(fig_velocity)
            plt.close(fig_power)
            plt.close(fig_force)
            plt.close(fig_force_FP)




### Helping code
#df_bench_freeweight = pd.read_csv(r'C:\D\Extra files\PhD\Benkpress\Frivekt + Muscle Lab\Bench Press FP_Mausehund Lasse_31-05-2023_16-12-49.csv', delimiter=';', decimal=",")
#print(df_bench_freeweight['Barbell velocity'])
#print(df_bench_freeweight.loc[0,'Barbell velocity'])
#type(df_bench_freeweight.loc[0,'Barbell velocity'])

