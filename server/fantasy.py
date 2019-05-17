
import pandas as pd
import json
from keras import models
import numpy as np
from keras.utils import normalize


from flask import Flask
from flask import jsonify

# Flask application
app = Flask(__name__)

API_KEY = '85d30890cf0c452a8129198a2ae07baa'

SEASON = 2018
WEEK = 17

AVG_NUMS = [1, 3, 5, 10]


# Fields we care about for predicting
# It is imperitive that the ordering of fields here matches those in generate_dataset.py

TRAINING_FIELDS_QB = ['FantasyPoints', 'PassingInterceptions', 'PassingTouchdowns', 'RushingAttempts', 'ReceivingYards', 'TwoPointConversionPasses',
    'TwoPointConversionRuns', 'TwoPointConversionReceptions', 'RushingYards', 'RushingTouchdowns', 'Fumbles', 'PassingAttempts', 'PassingCompletions']

TRAINING_FIELDS_OFFENSIVE = ['FantasyPoints', 'RushingAttempts', 'ReceivingYards', 'TwoPointConversionPasses', 'TwoPointConversionRuns', 'TwoPointConversionReceptions', 'RushingYards',
    'ReceivingTouchdowns', 'RushingTouchdowns', 'Fumbles', 'Receptions', 'ReceivingTargets', 'PuntReturns', 'KickReturns', 'PuntReturnYards', 'KickReturnYards', 'PuntReturnTouchdowns', 'KickReturnTouchdowns']

TRAINING_FIELDS_KICKER = ['FantasyPoints', 'ExtraPointsMade', 'FieldGoalsMade0to19', 'FieldGoalsMade20to29', 'FieldGoalsMade30to39', 'FieldGoalsMade40to49', 'FieldGoalsMade50Plus']

TRAINING_FIELDS_DEFENSE = ['FantasyPoints', 'SoloTackles', 'AssistedTackles', 'Sacks', 'Safeties', 'BlockedKicks', 'Interceptions', 'PassesDefended', 'FumblesRecovered',
    'FumblesForced', 'DefensiveTouchdowns', 'PuntReturnYards', 'KickReturnYards', 'FumbleReturnYards', 'InterceptionReturnYards', 'BlockedKickReturnYards', 'PuntReturnTouchdowns', 'KickReturnTouchdowns']


QB_DROP_COLS = ['TwoPointConversionPasses', 'TwoPointConversionRuns', 'TwoPointConversionReceptions']
OFFENSEIVE_DROP_COLS = ['TwoPointConversionPasses', 'TwoPointConversionRuns', 'TwoPointConversionReceptions', 'PuntReturns', 'KickReturns', 'PuntReturnYards', 'KickReturnYards', 'PuntReturnTouchdowns', 'KickReturnTouchdowns']
KICKER_DROP_COLS = ['FieldGoalsMade0to19', 'FieldGoalsMade20to29', 'FieldGoalsMade30to39']
DEFENSE_DROP_COLS = ['PuntReturnYards', 'KickReturnYards', 'FumbleReturnYards', 'InterceptionReturnYards', 'BlockedKickReturnYards', 'PuntReturnTouchdowns', 'KickReturnTouchdowns']


player_ids = pd.read_csv('../player_df.csv')

qb_model = models.load_model('../qb_player_fp_predict.h5')
offenseive_model = models.load_model('../offense_player_fp_predict.h5')
kicker_model = models.load_model('../kicker_player_fp_predict.h5')
defensive_model = models.load_model('../defense_player_fp_predict.h5')


def get_prediction(id):
    name, position, team, features = get_gameday_info(id)

    if position == 'QB':
        features.extend(get_qb_features(id))
        model = qb_model
    elif position in ['RB', 'WR']:
        features.extend(get_offense_features(id))
        model = offenseive_model
    elif position == 'K':
        features.extend(get_kicker_features(id))
        model = kicker_model
    elif position in ['LB', 'CB', 'S', 'SS', 'DT', 'DE', 'ILB', 'DE/LB', 'OLB', 'DL']:
        features.extend(get_defense_features(id))
        model = defensive_model
    else:
        return name, position, team, 0

    features = np.asarray(features)
    features = normalize(
        features.astype('float32'),
        axis=-1,
    )
    prediction = model.predict([features])

    return name, position, team, prediction[0][0]


def get_qb_features(player_id):
    features = get_all_weeks(player_id, TRAINING_FIELDS_QB, create_feature_df_qb)

    return features

def get_offense_features(player_id):
    features = get_all_weeks(player_id, TRAINING_FIELDS_OFFENSIVE, create_feature_df_offense)

    return features

def get_kicker_features(player_id):
    features = get_all_weeks(player_id, TRAINING_FIELDS_KICKER, create_feature_df_kicker)

    return features


def get_defense_features(player_id):
    features = get_all_weeks(player_id, TRAINING_FIELDS_DEFENSE, create_feature_df_defense)

    return features


def get_all_weeks(player_id, columns, create_feature_df):
    curr_week = WEEK - 1
    curr_season = SEASON

    max_weeks = AVG_NUMS[-1]
    games_found = 0
    attempts = 0

    all_weeks = None

    games = []

    while games_found < max_weeks or attempts > max_weeks * 2:
        attempts += 1
        stats = player_stats_by_week(curr_season, curr_week, player_id)

        curr_week -= 1
        if curr_week == 0:
            curr_week = 17
            curr_season -= 1

        if stats is None or stats.empty:
            continue

        games_found += 1

        # For some reason we get the same game 4 times, grab the first
        stats = stats[columns]
        games.append(stats)


    all_weeks = pd.concat(games, axis=1).transpose().reset_index()
    all_weeks = create_feature_df(all_weeks)
    num_games = len(all_weeks.index)

    features = [num_games]

    for num in AVG_NUMS:
        prev_games_set = all_weeks.tail(num)
        avgs = prev_games_set.mean(axis=0)
        features.extend(avgs.values[1:])

    season_stats = get_player_stats_season(curr_season, player_id)
    season_stats = season_stats[columns].fillna(0).div(season_stats['Played'], axis=0)
    season_stats = create_feature_df(season_stats)

    features.extend(season_stats.values[0])

    return features


def create_feature_df_qb(feature_df):
    # FantasyData fields are a bit different from Armchair, so fix that
    feature_df['Conversions'] =  feature_df['TwoPointConversionPasses'] + feature_df['TwoPointConversionRuns'] + feature_df['TwoPointConversionReceptions']

    feature_df = feature_df['FantasyPoints', 'PassingInterceptions', 'PassingTouchdowns', 'RushingAttempts', 'ReceivingYards', 'Conversions',
     'RushingYards', 'RushingTouchdowns', 'Fumbles', 'PassingAttempts', 'PassingCompletions']

    return feature_df


def create_feature_df_offense(feature_df):
    # FantasyData fields are a bit different from Armchair, so fix that
    feature_df['Conversions'] =  feature_df['TwoPointConversionPasses'] + feature_df['TwoPointConversionRuns'] + feature_df['TwoPointConversionReceptions']
    feature_df['Returns'] =  feature_df['PuntReturns'] + feature_df['KickReturns']
    feature_df['ReturnYardage'] =  feature_df['PuntReturnYards'] + feature_df['KickReturnYards']
    feature_df['ReturnTDs'] =  feature_df['PuntReturnTouchdowns'] + feature_df['KickReturnTouchdowns']

    feature_df = feature_df['FantasyPoints', 'RushingAttempts', 'ReceivingYards', 'Conversions', 'ReceivingTouchdowns', 'RushingYards',
     'RushingTouchdowns', 'Fumbles', 'Receptions', 'ReceivingTargets', 'Returns', 'ReturnYardage', 'ReturnTDs']

    return feature_df


def create_feature_df_kicker(feature_df):
    # FantasyData fields are a bit different from Armchair, so fix that
    feature_df['FieldGoalsMade0to40'] = feature_df['FieldGoalsMade0to19'] + feature_df['FieldGoalsMade20to29'] + feature_df['FieldGoalsMade30to39']
    feature_df = feature_df['FantasyPoints', 'ExtraPointsMade', 'FieldGoalsMade0to40', 'FieldGoalsMade40to49', 'FieldGoalsMade50Plus']

    return feature_df


def create_feature_df_defense(feature_df):
    # FantasyData fields are a bit different from Armchair, so fix that
    feature_df['ReturnYardage'] =  feature_df['PuntReturnYards'] + feature_df['KickReturnYards'] + feature_df['FumbleReturnYards'] \
        + feature_df['InterceptionReturnYards'] + feature_df['BlockedKickReturnYards'] + feature_df['KickReturnYards']
    feature_df['ReturnTDs'] = feature_df['PuntReturnTouchdowns'] + feature_df['KickReturnTouchdowns']

    feature_df = feature_df['FantasyPoints', 'SoloTackles', 'AssistedTackles', 'Sacks', 'Safeties', 'BlockedKicks', 'Interceptions', 'PassesDefended', 'FumblesRecovered',
     'FumblesForced', 'DefensiveTouchdowns', 'ReturnYardage', 'ReturnTDs']

    return feature_df


def get_gameday_info(id):
    player_url = f'https://api.sportsdata.io/v3/nfl/scores/json/Player/{id}?key={API_KEY}'

    player = pd.read_json(player_url, orient='records', typ='series')

    fname = player['FirstName']
    lname = player['LastName']
    jnum = player['Number']

    player_id_row = player_ids.loc[(player_ids['fname'] == fname) & (player_ids['lname'] == lname) & (player_ids['jnum'] == jnum)]
    player_id = player_id_row.index.data[0] + 2

    team = player['Team']

    with open("../team_dict.txt") as team_file:
        team_dict = json.load(team_file)

    team_id = team_dict[team]
    player_stats = player_stats_by_week(SEASON, WEEK, id)
    player_pos = player_stats['Position']

    home = 1 if player_stats['HomeOrAway'] == 'HOME' else 0

    opponent_abbr = player_stats['Opponent']
    opponent = team_dict[opponent_abbr]

    return f'{fname} {lname}', player_pos, team, [player_id, team_id, home, opponent, WEEK, SEASON]



def player_stats_by_week(season, week, player_id):
    stats_url = f'https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByPlayerID/{season}REG/{week}/{player_id}?key={API_KEY}'
    try:
        players_stats_df = pd.read_json(stats_url, orient='records', typ='series')
        return players_stats_df
    except:
        return None


def get_player_stats_season(season, player_id):
    stats_url = f'https://api.sportsdata.io/v3/nfl/stats/json/PlayerSeasonStatsByPlayerID/{season}REG/{player_id}?key={API_KEY}'
    try:
        players_stats_df = pd.read_json(stats_url)
        return players_stats_df
    except:
        return None



@app.route('/')
def main():
    return 'Server for Fantasy NN.'


@app.route('/player/<player_id>')
def player(player_id):
    try:
        name, position, team, points = get_prediction(player_id)
    except:
        name = 'Unknown'
        position = '--'
        team = '--'
        points = 0
    info = [{
        "PlayerName": name,
        "PlayerPos": position,
        "PlayerTeam": team,
        "PlayerPointsProjected": points,
        }]
    return jsonify(info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)


