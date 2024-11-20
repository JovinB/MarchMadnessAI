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

def get_averaged_reg_season_data(data_tldr):
    # TODO: make work across seasons
    years = [2022]
    data = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    data = data.loc[data['Season'].isin(years)]

    averaged_data = []
    team_IDs = []

    # data: WFGM,WFGA,WFGM3,WFGA3,WFTM,WFTA,WOR,WDR,WAst,WTO,WStl,WBlk,WPF (again for the losing team, but with prefix 'L' instead of 'W')

    for game in data.iterrows():
        game = game[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data
        team_1_ID = game['WTeamID']
        team_2_ID = game['LTeamID']

        if team_1_ID not in team_IDs:
            team_IDs.append(team_1_ID)

            averaged_data.append([1,team_1_ID,int(game['WFGM']),int(game['WFGA']),int(game['WFGM3']),int(game['WFGA3']),
                                  int(game['WFTM']),int(game['WFTA']),int(game['WOR']),int(game['WDR']),int(game['WAst']),
                                  int(game['WTO']),int(game['WStl']),int(game['WBlk']),int(game['WPF'])])
        else:
            for i, team_ID in enumerate(team_IDs):
                if team_ID == team_1_ID:
                    averaged_data[i] = [averaged_data[i][0]+1,averaged_data[i][1],
                                        averaged_data[i][2]+int(game['WFGM']),averaged_data[i][3]+int(game['WFGA']),
                                        averaged_data[i][4]+int(game['WFGM3']),averaged_data[i][5]+int(game['WFGA3']),
                                        averaged_data[i][6]+int(game['WFTM']),averaged_data[i][7]+int(game['WFTA']),
                                        averaged_data[i][8]+int(game['WOR']),averaged_data[i][9]+int(game['WDR']),
                                        averaged_data[i][10]+int(game['WAst']),averaged_data[i][11]+int(game['WTO']),
                                        averaged_data[i][12]+int(game['WStl']),averaged_data[i][13]+int(game['WBlk']),
                                        averaged_data[i][14]+int(game['WPF'])]
                    break
        
        if team_2_ID not in team_IDs:
            team_IDs.append(team_2_ID)

            averaged_data.append([1,team_2_ID,int(game['LFGM']),int(game['LFGA']),int(game['LFGM3']),int(game['LFGA3']),
                                  int(game['LFTM']),int(game['LFTA']),int(game['LOR']),int(game['LDR']),int(game['LAst']),
                                  int(game['LTO']),int(game['LStl']),int(game['LBlk']),int(game['LPF'])])
        else:
            for i, team_ID in enumerate(team_IDs):
                if team_ID == team_2_ID:
                    averaged_data[i] = [averaged_data[i][0]+1,averaged_data[i][1],
                                        averaged_data[i][2]+int(game['LFGM']),averaged_data[i][3]+int(game['LFGA']),
                                        averaged_data[i][4]+int(game['LFGM3']),averaged_data[i][5]+int(game['LFGA3']),
                                        averaged_data[i][6]+int(game['LFTM']),averaged_data[i][7]+int(game['LFTA']),
                                        averaged_data[i][8]+int(game['LOR']),averaged_data[i][9]+int(game['LDR']),
                                        averaged_data[i][10]+int(game['LAst']),averaged_data[i][11]+int(game['LTO']),
                                        averaged_data[i][12]+int(game['LStl']),averaged_data[i][13]+int(game['LBlk']),
                                        averaged_data[i][14]+int(game['LPF'])]
                    break

    for i in range(len(averaged_data)):
        total_games = averaged_data[i][0]
        averaged_data[i] = [averaged_data[i][1],averaged_data[i][2]/total_games,averaged_data[i][3]/total_games,
                            averaged_data[i][4]/total_games,averaged_data[i][5]/total_games,averaged_data[i][6]/total_games,
                            averaged_data[i][7]/total_games,averaged_data[i][8]/total_games,averaged_data[i][9]/total_games,
                            averaged_data[i][10]/total_games,averaged_data[i][11]/total_games,averaged_data[i][12]/total_games,
                            averaged_data[i][13]/total_games,averaged_data[i][14]/total_games]

    return averaged_data

def get_data(start_year, end_year):
    reg_season_data = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    seed_data = pd.read_csv("data/MNCAATourneySeeds.csv")
    tourney_result_data = pd.read_csv("data/MNCAATourneyCompactResults.csv")

    years = []
    year = start_year
    while not year == end_year + 1:
        years.append(year)
        year += 1

    reg_season_data_subset = get_averaged_reg_season_data(reg_season_data.loc[reg_season_data['Season'].isin(years)])
    seed_data_subset = seed_data.loc[seed_data['Season'].isin(years)]
    tourney_data_subset = tourney_result_data.loc[tourney_result_data['Season'].isin(years)]

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

testing_data = get_averaged_reg_season_data([1])

for row in testing_data:
    if row[0] == 1101:
        print(row)
        break