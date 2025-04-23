import requests
import pandas as pd
import numpy as np
from constants import DATA_TAGS, COLUMNS_OF_INTEREST
#data of interest

date_format = "%Y-%m-%d"
start_date_str = '2024-01-01'
end_date_str = '2025-04-17'

data_dict = {}
for item in DATA_TAGS:
    data_dict[item] = []

for data in DATA_TAGS:
    request = requests.get(f'https://sky.coflnet.com/api/bazaar/{data}/history?start={start_date_str}T12%3A00%3A00.004Z&end={end_date_str}T12%3A00%3A00.004Z')
    if request.status_code == 200:
        resp = request.json()
        # Process the data
        data_dict[data] = resp + data_dict[data]
    else:
        print(f"Error: {request.status_code}")

#Drop NaN's and reorganize for Pandas
for data in DATA_TAGS:
    data_dict[data] = pd.DataFrame(data_dict[data]).dropna(axis = 0)

#Rewrite timestamps as YYYY-MM-DD
for data in DATA_TAGS:
    data_dict[data]["timestamp"] = data_dict[data]["timestamp"].apply(lambda datestr : datestr[:10])
#Get common timestamps
common_times = data_dict[DATA_TAGS[0]]['timestamp'].unique()
for data in DATA_TAGS:
    common_times = np.intersect1d(common_times, data_dict[data]['timestamp'].values)
#Flush out all rows that don't correspond with a common time
for data in DATA_TAGS:
    data_dict[data] = data_dict[data][data_dict[data]['timestamp'].isin(common_times)]
#Get unique dates
for data in DATA_TAGS:
    data_dict[data] = data_dict[data].drop_duplicates(subset = ['timestamp'], keep = 'first')
#Sort by datetime
for data in DATA_TAGS:
    data_dict[data] = data_dict[data].sort_values(by='timestamp')
flattened_data_dict = {}
#flatten to convert to Pandas Dataframe
for data in DATA_TAGS:
    for column in COLUMNS_OF_INTEREST + ['timestamp']:
        flattened_data_dict[f'{data}_{column}'] = data_dict[data][column].values

flattened_data_df = pd.DataFrame(flattened_data_dict)
flattened_data_df.to_pickle("data/raw_datafile")

