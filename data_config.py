import pandas as pd


TEAM_DIRECTORY = {}

# create directory of key/value pairs with TeamID/TeamName
team_data = pd.read_csv("data/MTeams.csv")
for row in team_data.iterrows():
    row = row[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data
    TEAM_DIRECTORY[row['TeamID']] = row['TeamName']

