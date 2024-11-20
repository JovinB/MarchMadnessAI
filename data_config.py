import pandas as pd
import random
import numpy as np

def establish_team_directory():
    team_directory = {}

    # create directory of key/value pairs with TeamID/TeamName
    team_data = pd.read_csv("data/MTeams.csv")
    for row in team_data.iterrows():
        row = row[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data
        team_directory[row['TeamID']] = row['TeamName']

    return team_directory


def get_data(start_year, end_year):
    seed_data = pd.read_csv("data/MNCAATourneySeeds.csv")
    tourney_data = pd.read_csv("data/MNCAATourneyCompactResults.csv")

    years = []
    year = start_year
    while not year == end_year + 1:
        years.append(year)
        year += 1

    seed_data_subset = seed_data.loc[seed_data['Season'].isin(years)]
    tourney_data_subset = tourney_data.loc[tourney_data['Season'].isin(years)]

    data = []

    for game in tourney_data_subset.iterrows():
        game = game[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data
        game_data = []

        w_seed = seed_data_subset.loc[(seed_data_subset['TeamID'] == game['WTeamID']) & (seed_data_subset['Season'] == game['Season'])]['Seed'].values[0]
        l_seed = seed_data_subset.loc[(seed_data_subset['TeamID'] == game['LTeamID']) & (seed_data_subset['Season'] == game['Season'])]['Seed'].values[0]

        # we do not want the conference identifier on the seed
        w_seed = w_seed.replace('W','')
        l_seed = l_seed.replace('W','')
        w_seed = w_seed.replace('X','')
        l_seed = l_seed.replace('X','')
        w_seed = w_seed.replace('Y','')
        l_seed = l_seed.replace('Y','')
        w_seed = w_seed.replace('Z','')
        l_seed = l_seed.replace('Z','')
        w_seed = w_seed.replace('a','')
        l_seed = l_seed.replace('a','')
        w_seed = w_seed.replace('b','')
        l_seed = l_seed.replace('b','')

        if random.random() < 0.5: # randomly choose to order the data as winner, loser, 1 (label) or loser, winner, 0 (label)
            game_data.append(int(w_seed))
            game_data.append(int(l_seed))
            game_data.append(1)
        else:
            game_data.append(int(l_seed))
            game_data.append(int(w_seed))
            game_data.append(0)

        data.append(game_data)
    
    return np.array(data)

get_data(2021, 2022)