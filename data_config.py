import pandas as pd
import random
import numpy as np
import csv


def convert_team_name_format(team_name):
    team_name = team_name.replace(".","")
    match team_name:
        case "Saint Mary's":
            return "St Mary's CA"
        case "Florida Atlantic":
            return "FL Atlantic"
        case "Charleston":
            return "Col Charleston"
        case "Kent St":
            return "Kent"
        case "Kennesaw St":
            return "Kennesaw"
        case "Northern Kentucky":
            return "N Kentucky"
        case "Eastern Kentucky":
            return "E Kentucky"
        case "Southern Illinois":
            return "S Illinois"
        case "Illinois Chicago":
            return "IL Chicago"
        case "Boston University":
            return "Boston Univ"
        case "Milwaukee":
            return "WI Milwaukee"
        case "Troy St":
            return "Troy"
        case "Central Connecticut":
            return "Central Conn"
        case "Saint Joseph's":
            return "St Joseph's PA"
        case "Central Michigan":
            return "C Michigan"
        case "East Tennessee St":
            return "ETSU"
        case "South Carolina St":
            return "S Carolina St"
        case "Texas Southern":
            return "TX Southern"
        case "Western Michigan":
            return "W Michigan"
        case "Louisiana Lafayette":
            return "Louisiana"
        case "Eastern Washington":
            return "E Washington"
        case "Monmouth":
            return "Monmouth NJ"
        case "UTSA":
            return "UT San Antonio"
        case "George Washington":
            return "G Washington"
        case "Southeastern Louisiana":
            return "SE Louisiana"
        case "Fairleigh Dickinson":
            return "F Dickinson"
        case "Southern":
            return "Southern Univ"
        case "Northwestern St":
            return "Northwestern LA"
        case "Western Kentucky":
            return "WKU"
        case "Texas A&M Corpus Chris":
            return "TAM C. Christi"
        case "Albany":
            return "SUNY Albany"
        case "North Dakota St":
            return "N Dakota St"
        case "American":
            return "American Univ"
        case "Arkansas Pine Bluff":
            return "Ark Pine Bluff"
        case "Northern Colorado":
            return "N Colorado"
        case "Saint Peter's":
            return "St Peter's"
        case "Arkansas Little Rock":
            return "Ark Little Rock"
        case "Saint Louis":
            return "St Louis"
        case "South Dakota St":
            return "S Dakota St"
        case "Mississippi Valley St":
            return "MS Valley St"
        case "Middle Tennessee":
            return "MTSU"
        case "Florida Gulf Coast":
            return "FL Gulf Coast"
        case "North Carolina A&T":
            return "NC A&T"
        case "Stephen F Austin":
            return "SF Austin"
        case "North Carolina Central":
            return "NC Central"
        case "Mount St Mary's":
            return "Mt St Mary's"
        case "Coastal Carolina":
            return "Coastal Car"
        case "Cal St Bakersfield":
            return "CS Bakersfield"
        case "Green Bay":
            return "WI Green Bay"
        case "Loyola Chicago":
            return "Loyola-Chicago"
        case "College of Charleston":
            return "Col Charleston"
        case "Cal St Fullerton":
            return "CS Fullerton"
        case "Prairie View A&M":
            return "Prairie View"
        case "Abilene Christian":
            return "Abilene Chr"
        case _:
            return team_name

def establish_team_directory():
    team_directory = {}

    # create directory of key/value pairs with TeamName/TeamID
    team_data = pd.read_csv("data/MTeams.csv")
    for row in team_data.iterrows():
        row = row[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data
        team_directory[row['TeamID']] = row['TeamName']

    return team_directory

def clean_seed(seed):  # removes conference identifiers from the team's seed
    seed = seed.replace('W','')
    seed = seed.replace('X','')
    seed = seed.replace('Y','')
    seed = seed.replace('Z','')
    seed = seed.replace('a','')
    seed = seed.replace('b','')

    return seed

def count_numbers_in_string(string):
    numbers_count = 0
    for char in string:
        if char.isdigit():
            numbers_count += 1
    
    return numbers_count

def get_team_data(data, teamID, season):
    for team in data:
        if team[0] == teamID and team[1] == season:
            return team[2:]
    return None

def get_kenpom_team_data(team_directory, data, teamID, season):
    for row in data:
        if row[0] == team_directory[teamID] and row[1] == season:
            return row[2:]

    return None

def get_averaged_reg_season_data(data):
    averaged_data = []
    teams = []

    # data: WFGM,WFGA,WFGM3,WFGA3,WFTM,WFTA,WOR,WDR,WAst,WTO,WStl,WBlk,WPF (again for the losing team, but with prefix 'L' instead of 'W')

    for game in data.iterrows():
        game = game[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data
        team_1_ID = game['WTeamID']
        team_2_ID = game['LTeamID']
        season = game['Season']

        if (team_1_ID, season) not in teams:
            teams.append((team_1_ID, season))

            averaged_data.append([1,team_1_ID,season,int(game['WFGM']),int(game['WFGA']),int(game['WFGM3']),int(game['WFGA3']),
                                  int(game['WFTM']),int(game['WFTA']),int(game['WOR']),int(game['WDR']),int(game['WAst']),
                                  int(game['WTO']),int(game['WStl']),int(game['WBlk']),int(game['WPF'])])
        else:
            for i, team in enumerate(teams):
                if team[0] == team_1_ID and team[1] == season:
                    averaged_data[i] = [averaged_data[i][0]+1,averaged_data[i][1],averaged_data[i][2],
                                        averaged_data[i][3]+int(game['WFGM']),averaged_data[i][4]+int(game['WFGA']),
                                        averaged_data[i][5]+int(game['WFGM3']),averaged_data[i][6]+int(game['WFGA3']),
                                        averaged_data[i][7]+int(game['WFTM']),averaged_data[i][8]+int(game['WFTA']),
                                        averaged_data[i][9]+int(game['WOR']),averaged_data[i][10]+int(game['WDR']),
                                        averaged_data[i][11]+int(game['WAst']),averaged_data[i][12]+int(game['WTO']),
                                        averaged_data[i][13]+int(game['WStl']),averaged_data[i][14]+int(game['WBlk']),
                                        averaged_data[i][15]+int(game['WPF'])]
                    break
        
        if (team_2_ID, season) not in teams:
            teams.append((team_2_ID, season))

            averaged_data.append([1,team_2_ID,season,int(game['LFGM']),int(game['LFGA']),int(game['LFGM3']),int(game['LFGA3']),
                                  int(game['LFTM']),int(game['LFTA']),int(game['LOR']),int(game['LDR']),int(game['LAst']),
                                  int(game['LTO']),int(game['LStl']),int(game['LBlk']),int(game['LPF'])])
        else:
            for i, team in enumerate(teams):
                if team[0] == team_2_ID and team[1] == season:
                    averaged_data[i] = [averaged_data[i][0]+1,averaged_data[i][1],averaged_data[i][2],
                                        averaged_data[i][3]+int(game['LFGM']),averaged_data[i][4]+int(game['LFGA']),
                                        averaged_data[i][5]+int(game['LFGM3']),averaged_data[i][6]+int(game['LFGA3']),
                                        averaged_data[i][7]+int(game['LFTM']),averaged_data[i][8]+int(game['LFTA']),
                                        averaged_data[i][9]+int(game['LOR']),averaged_data[i][10]+int(game['LDR']),
                                        averaged_data[i][11]+int(game['LAst']),averaged_data[i][12]+int(game['LTO']),
                                        averaged_data[i][13]+int(game['LStl']),averaged_data[i][14]+int(game['LBlk']),
                                        averaged_data[i][15]+int(game['LPF'])]
                    break

    for i in range(len(averaged_data)):
        total_games = averaged_data[i][0]
        total_games = 1  # for testing purposes -- gives totals instead of averages
        averaged_data[i] = [averaged_data[i][1],averaged_data[i][2],averaged_data[i][3]/total_games,
                            averaged_data[i][4]/total_games,averaged_data[i][5]/total_games,averaged_data[i][6]/total_games,
                            averaged_data[i][7]/total_games,averaged_data[i][8]/total_games,averaged_data[i][9]/total_games,
                            averaged_data[i][10]/total_games,averaged_data[i][11]/total_games,averaged_data[i][12]/total_games,
                            averaged_data[i][13]/total_games,averaged_data[i][14]/total_games,averaged_data[i][15]/total_games]

    return averaged_data


def get_kenpom_data(start_year, end_year):
    years = []
    year = start_year
    while not year == end_year + 1:
        if year == 2020:
            year += 1
            continue
        years.append(year)
        year += 1

    kenpom_data = []

    for year in years:
        file = open(f"data/kenpom_{year}.txt", 'r')

        file_reader = csv.reader(file, delimiter="\t")
                        
        for line in file_reader:
            seed_num_length = count_numbers_in_string(line[1])

            # if the team is not seeded, skip its line of data
            if seed_num_length > 0:
                wins, losses = line[3].split("-")
                win_loss_ratio = int(wins) / int(losses)

                # only include desired data
                data_row = []
                team_name = convert_team_name_format(line[1][:-(1+seed_num_length)])
                data_row.append(team_name)
                data_row.append(year)
                data_row.append(win_loss_ratio) # engineered feature
                data_row.append(float(line[4]))
                data_row.append(float(line[5]))
                data_row.append(float(line[7]))
                data_row.append(float(line[9]))
                data_row.append(float(line[11]))
                data_row.append(float(line[13]))
                data_row.append(float(line[15]))
                data_row.append(float(line[17]))
                data_row.append(float(line[19]))

                kenpom_data.append(data_row)
    
    return kenpom_data


def get_data(start_year, end_year, dataset="Regular"): # dataset can also be 'KenPom' or 'Both'
    TEAM_DIRECTORY = establish_team_directory()
    reg_season_data = pd.read_csv("data/MRegularSeasonDetailedResults.csv")
    seed_data = pd.read_csv("data/MNCAATourneySeeds.csv")
    tourney_result_data = pd.read_csv("data/MNCAATourneyCompactResults.csv")

    years = []
    year = start_year
    while not year == end_year + 1:
        if year == 2008 or year == 2020: # kenpom is having trouble with 2008, and 2020 there is no data
            year += 1
            continue
        years.append(year)
        year += 1

    seed_data_subset = seed_data.loc[seed_data['Season'].isin(years)]
    tourney_data_subset = tourney_result_data.loc[tourney_result_data['Season'].isin(years)]

    if dataset != "KenPom":
        reg_season_data_subset = get_averaged_reg_season_data(reg_season_data.loc[reg_season_data['Season'].isin(years)])
    if dataset == "KenPom" or dataset == "Both":
        kenpom_data_subset = get_kenpom_data(start_year, end_year)

    data = []

    for game in tourney_data_subset.iterrows():
        game_data = None
        game = game[1] # iterrows() returns a (index, Series) pair where the 'Series' contains the data

        w_seed = seed_data_subset.loc[(seed_data_subset['TeamID'] == game['WTeamID']) & (seed_data_subset['Season'] == game['Season'])]['Seed'].values[0]
        l_seed = seed_data_subset.loc[(seed_data_subset['TeamID'] == game['LTeamID']) & (seed_data_subset['Season'] == game['Season'])]['Seed'].values[0]

        # we do not want the conference identifier on the seed
        w_seed = clean_seed(w_seed)
        l_seed = clean_seed(l_seed)

        if dataset == "Regular":
            w_team_data = get_team_data(reg_season_data_subset, game['WTeamID'], game['Season'])
            l_team_data = get_team_data(reg_season_data_subset, game['LTeamID'], game['Season'])

        elif dataset == "KenPom":
            w_team_data = get_kenpom_team_data(TEAM_DIRECTORY, kenpom_data_subset, game['WTeamID'], game['Season'])
            l_team_data = get_kenpom_team_data(TEAM_DIRECTORY, kenpom_data_subset, game['LTeamID'], game['Season'])

        elif dataset == "Both":
            w_team_data = get_team_data(reg_season_data_subset, game['WTeamID'], game['Season']) + get_kenpom_team_data(TEAM_DIRECTORY, kenpom_data_subset, game['WTeamID'], game['Season'])
            l_team_data = get_team_data(reg_season_data_subset, game['LTeamID'], game['Season']) + get_kenpom_team_data(TEAM_DIRECTORY, kenpom_data_subset, game['LTeamID'], game['Season'])

        if random.random() < 0.5: # randomly choose to order the data as winner, loser, 1 (label) or loser, winner, 0 (label)
            try:
                game_data = [int(w_seed)] + w_team_data + [int(l_seed)] + l_team_data + [1]
            except:
                print(f"{game['Season']}, {game['WTeamID']}, {game['LTeamID']}")
                print(f"Game data: {game_data}")
        else:
            try:
                game_data = [int(l_seed)] + l_team_data + [int(w_seed)] + w_team_data + [0]
            except:
                print(f"{game['Season']}, {game['WTeamID']}, {game['LTeamID']}")
                print(f"Game data: {game_data}")

        if game_data != None:
            data.append(game_data)
        else:
            print('Skipped appending')
    
    return np.array(data)