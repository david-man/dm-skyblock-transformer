import pandas as pd
from datetime import datetime, timedelta
from constants import DATA_TAGS, DATE_FORMAT, COLUMNS_OF_INTEREST, PREDICTION_HORIZON, LABEL_COLUMN
import numpy as np
flattened_data_df = pd.read_pickle("data/raw_datafile")
#create official timestamp index
timestep_columns_to_drop = [f'{data}_timestamp' for data in DATA_TAGS]
flattened_data_df['timestamp'] = flattened_data_df[f'{DATA_TAGS[0]}_timestamp']
flattened_data_df = flattened_data_df.drop(timestep_columns_to_drop, axis = 1)

data_length = PREDICTION_HORIZON + 1
#find the streaks of days that we can use for training
streaks = []
for timestep in flattened_data_df['timestamp']:
    next_timesteps = []
    current_day = datetime.strptime(timestep, DATE_FORMAT)
    for i in range(1, data_length):#get the next HORIZON days from this timestep in string format
        next_day = current_day + timedelta(days = i)
        next_day_string = next_day.strftime(DATE_FORMAT)
        next_timesteps.append(next_day_string)
    #check if the dataframe has data on every timestep in the horizon. if it does, data is usable
    all_contained = True
    for time in next_timesteps:
        all_contained &= (time in flattened_data_df['timestamp'].values)
    if(all_contained):
        streaks.append([timestep] + next_timesteps)
#set index by timestamp for easy access
flattened_data_df = flattened_data_df.set_index('timestamp')
#organize the dataset and the labels
dataset = []
labels = []
for streak in streaks:
    streak_data = []
    for time in streak:
        if(time == streak[-1]):
            label = []
            for data in DATA_TAGS:
                #normalization parameter at the beginning of the streak
                norm_param = flattened_data_df.loc[streak[0]][f'{data}_{LABEL_COLUMN}']

                label.append(flattened_data_df.loc[time][f'{data}_{LABEL_COLUMN}'] / norm_param)
            labels.append(np.array(label))
        else:
            time_data = []
            for data in DATA_TAGS:
                tag_specific_data = []
                for column in COLUMNS_OF_INTEREST:
                    
                    string = f"{data}_{column}"
                    #normalization parameter at the beginning of the streak
                    norm_param = flattened_data_df.loc[streak[0]][string]
                    tag_specific_data.append(flattened_data_df.loc[time][string] / norm_param)
                time_data.append(np.array(tag_specific_data))
            streak_data.append(np.array(time_data))
    dataset.append(np.array(streak_data))
dataset = np.array(dataset)
labels = np.array(labels)
dataset = dataset.swapaxes(1, 2)#swap it so stocks are on the outside
if(len(dataset.shape) == 3):
    dataset = np.expand_dims(dataset, axis = -1)#make sure the last axis isn't unsqueezed -> B x |S| x t x F
labels = np.expand_dims(labels, axis = -1)#make it B x |S| x 1
np.save('data/training_dataset', dataset)
np.save('data/training_labels', labels)