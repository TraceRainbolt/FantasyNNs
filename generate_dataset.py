import sys
import json

import pandas
import numpy as np


# All training data

schedule_df = pandas.read_csv('data/SCHEDULE.csv')
player_df = pandas.read_csv('data/PLAYER.csv')
games_df = pandas.read_csv('data/GAME.csv')
offense_df = pandas.read_csv('data/OFFENSE.csv')
team_df = pandas.read_csv('data/TEAM.csv')
kicker_df = pandas.read_csv('data/KICKER.csv')
defense_df = pandas.read_csv('data/DEFENSE.csv')

rbs_and_wrs = player_df.loc[(player_df['pos1'].isin(['RB', 'WR']))]
qbs = player_df.loc[(player_df['pos1'].isin(['QB']))]

# Number of weeks in an NFL regular season
REG_SEASON_WEEKS = 17

# Fields we care about for training
TRAINING_FIELDS_QB = ['fp', 'ints', 'tdp', 'ra', 'recy', 'conv', 'ry', 'tdr', 'fuml', 'pa', 'pc']
TRAINING_FIELDS_OFFENSIVE = ['fp', 'ra',  'recy', 'conv', 'tdrec', 'ry', 'tdr', 'fuml', 'rec', 'trg', 'ret', 'rety', 'tdret']
TRAINING_FIELDS_KICKER = ['fp', 'pat', 'fgs', 'fgm', 'fgl']
TRAINING_FIELDS_DEFENSE = ['fp', 'solo', 'comb', 'sck', 'saf', 'blk', 'ints', 'pdef', 'frcv', 'forc', 'tdd', 'rety', 'tdret']

FIELDS_MAP = {
    'qb': TRAINING_FIELDS_QB,
    'offense': TRAINING_FIELDS_OFFENSIVE,
    'kicker': TRAINING_FIELDS_KICKER,
    'defense': TRAINING_FIELDS_DEFENSE,
}

# Which time lengths to take averages of
AVG_NUMS = [1, 3, 5, 10]

# How many games a player must have played to use them in training
MIN_GAMES = 1

# Minumum average points to consider player in training
MIN_FP_AVG = 0

# Used to convert team name to number
team_dict = team_df['tname'].to_dict()
player_dict = player_df['player'].to_dict()

# Reverse the mapping
team_dict = {v: k for k, v in team_dict.items()}
player_dict = {v: k for k, v in player_dict.items()}

# # Server needs dictionaries
# with open('team_dict.txt', 'w') as team_file:
#     json.dump(team_dict, team_file)

player_lookup = player_df[['player', 'fname', 'lname', 'jnum']].to_csv('player_df.csv')


def generate_dataset(position):
    all_weeks = None
    dataset = []
    for _, week in schedule_df.groupby(['seas', 'wk']):
        week_num = week['wk'].iloc[0]
        seas_num = week['seas'].iloc[0]

        if seas_num == 2018 or week_num > REG_SEASON_WEEKS:
            continue

        weekly_stats = get_weekly_stats(week, position)

        if all_weeks is None:
            all_weeks = weekly_stats
        else:
            all_weeks = all_weeks.append(weekly_stats)

    for _, player in all_weeks.groupby('player'):
        for index, week in player.iterrows():
            features = get_features(week, player, AVG_NUMS, position)
            if features is not None:
                dataset.append(features)

    return dataset


def get_features(week, player_games, nums, position):
    current_week = week['week']
    current_season = week['season']

    previous_games = player_games.loc[(player_games['week'] < current_week) & (player_games['season'] <= current_season)]

    num_games = len(previous_games.index)

    if num_games < MIN_GAMES:
        return

    fields = FIELDS_MAP[position]

    features = [week['fp'], week['player_id'], week['team'], week['home'], week['opponent'],
            week['week'], week['season'], num_games]

    for num in nums:
        prev_games_set = previous_games.tail(num)
        avgs = prev_games_set[fields].mean(axis=0)
        features.extend(avgs.values)

    season_avgs = player_games[fields].loc[(player_games['season'] == current_season)].mean(axis=0)
    features.extend(season_avgs)

    return features


def get_weekly_stats(week, position):
    week_num = week['wk'].iloc[0]
    seas_num = week['seas'].iloc[0]
    data_df = None
    stats_df = None

    for index, game in week.iterrows():
        gid = game['gid']

        if position == 'offense':
            new_players = offense_df.loc[offense_df['gid'] == gid]
            new_players = new_players.loc[new_players['player'].isin(rbs_and_wrs['player'])]
        elif position == 'qb':
            new_players = offense_df.loc[offense_df['gid'] == gid]
            new_players = new_players.loc[new_players['player'].isin(qbs['player'])]
        elif position == 'kicker':
            new_players = kicker_df.loc[kicker_df['gid'] == gid]
        elif position == 'defense':
            new_players = defense_df.loc[defense_df['gid'] == gid]


        fields = ['player'] + FIELDS_MAP[position]
        new_data = new_players[fields]
        new_data = new_data.set_index('player')

        if data_df is None:
            data_df = new_data
        else:
            data_df = data_df.add(new_data, fill_value=0)

        teams = new_players[['player', 'team']]
        teams = teams.set_index('player', drop=False)
        visiting = game['v']
        home = game['h']

        opponents = teams['team'].apply(lambda x: team_dict[home] if x == visiting else team_dict[visiting])
        team_nums = teams['team'].apply(lambda x: team_dict[x])
        home = teams['team'].apply(lambda x: 0 if x == visiting else 1)
        player_ids = teams['player'].apply(lambda x: player_dict[x])

        teams['opponent'] = opponents
        teams['team'] = team_nums
        teams['home'] = home
        teams['player_id'] = player_ids

        if stats_df is None:
            stats_df = teams
        else:
            stats_df = stats_df.append(teams)

    data_df = data_df.merge(stats_df, how='left', on='player')
    data_df['week'] = week_num
    data_df['season'] = seas_num
    return data_df


if __name__ == '__main__':
    position = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out_file = f'{position}_dataset.npy'

    dataset = generate_dataset(position)
    np.save(out_file, dataset)
    print(f'Successfully generated dataset with {len(dataset)} entries.')


# from flask import Flask
# from flask import jsonify

# app = Flask(__name__)


# @app.route('/')
# def main():
#     return 'Server for Fantasy NN.'


# @app.route('/player/<player_id>')
# def player(player_id):
#     point_predict = get_prediction(player_id)
#     info = [{
#         "PlayerName": None,
#         "PlayerPos": None,
#         "PlayerTeam": None,
#         "PlayerPointsProjected": point_predict,
#         }]
#     return jsonify(info)

# def get_prediction(player_id):
#     points = hash(player_id) % 100 / 5.0
#     return points

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=80)

description "uWSGI server instance configured to serve myproject"

start on runlevel [2345]
stop on runlevel [!2345]

setuid user
setgid www-data

env PATH=/home/user/fantasy/fantasy/bin
chdir /home/user/fantasy/fantasy.ini
exec uwsgi --ini fantasy.ini

Description=uWSGI instance to serve myproject
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/fantasy
Environment="PATH=/home/ubuntu/fantasy/fantasy/bin"
ExecStart=/home/ubuntu/fantasy/fantasy/bin/uwsgi --ini fantasy.ini

[Install]
WantedBy=multi-user.target

---
[uwsgi]
module = wsgi:app

master = true
processes = 5

socket = fantasy.sock
chmod-socket = 660
vacuum = true

die-on-term = true

server {
    listen 80;
    server_name http://18.191.177.118/ 18.191.177.118;

    location / {
        include uwsgi_params;
        uwsgi_pass unix:/home/sammy/myproject/myproject.sock;
    }
}