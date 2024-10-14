# directory = "/mnt/d/AIHackercup/USACO/data/corpuses/cpbook_v2.json"
import json
import pandas as pd 
#load the json file and save to pd.dataframe
#extract the section names

# Load the JSON file and extract the data into a pandas DataFrame
directory = "/mnt/d/AIHackercup/USACO/data/corpuses/cpbook_v2.json"
directory = "/mnt/d/AIHackercup/USACO/data/corpuses/cp_handbook.json"

with open(directory, 'r') as file:
    data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.json_normalize(data)

# Extract section names (if they are part of the data structure)
section_names = df.columns.tolist()
