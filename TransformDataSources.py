import pandas as pd
from datetime import datetime

def CombinePlayerDataSources(SummaryDF: pd.DataFrame, MiscellaneousDF: pd.DataFrame, PP1DF: pd.DataFrame) -> pd.DataFrame:

    MergedDF = SummaryDF.merge(MiscellaneousDF, on='playerId')
    MergedDF['pp1_status'] = MergedDF['skaterFullName_x'].isin(PP1DF['player_name'])
    MergedDF['fetch_date'] = datetime.today().date()

    CopyDF =  MergedDF[['fetch_date',
                        'playerId',
                        'skaterFullName_x',
                        'teamAbbrevs_x',
                        'gamesPlayed_x',
                        'goals',
                        'assists',
                        'plusMinus',
                        'ppPoints',
                        'shots',
                        'hits',
                        'blockedShots',
                        'pp1_status',
                        'timeOnIcePerGame_x']].copy()
    
    CopyDF.rename(columns={
        'playerId' : 'player_id',
        'skaterFullName_x': 'player_full_name',
        'teamAbbrevs_x': 'team_abrev',
        'gamesPlayed_x': 'games_played',
        'plusMinus': 'plus_minus',
        'ppPoints': 'pp_points',
        'blockedShots': 'blocked_shots'
    }, inplace=True)

    # If a player was traded there might be more than one team in the team_abrev column
    CopyDF['team_abrev'] = CopyDF['team_abrev'].apply(lambda x: x[-3:] if len(x) > 3 else x)

    return CopyDF[['player_id', 'fetch_date', 'player_full_name', 'team_abrev', 'games_played', 'goals', 'assists', 'plus_minus', 'pp_points', 'shots', 'hits', 'blocked_shots', 'pp1_status']]


def CleanTeamDataSource(TeamDF: pd.DataFrame) -> pd.DataFrame:
    TeamDF.to_csv('TeamDF.csv', index=False)

    TeamDF['fetch_date'] = datetime.today().date()

    team_abbreviations = {
        'Ottawa Senators': 'OTT',
        'Utah Hockey Club': 'UTA',
        'Carolina Hurricanes': 'CAR',
        'Vegas Golden Knights': 'VGK',
        'Edmonton Oilers': 'EDM',
        'Dallas Stars': 'DAL',
        'Anaheim Ducks': 'ANA',
        'New York Rangers': 'NYR',
        'Detroit Red Wings': 'DET',
        'Montréal Canadiens': 'MTL',
        'Columbus Blue Jackets': 'CBJ',
        'Chicago Blackhawks': 'CHI',
        'Nashville Predators': 'NSH',
        'St. Louis Blues': 'STL',
        'New Jersey Devils': 'NJD',
        'Minnesota Wild': 'MIN',
        'Buffalo Sabres': 'BUF',
        'Winnipeg Jets': 'WPG',
        'Philadelphia Flyers': 'PHI',
        'Vancouver Canucks': 'VAN',
        'Boston Bruins': 'BOS',
        'Washington Capitals': 'WSH',
        'San Jose Sharks': 'SJS',
        'Toronto Maple Leafs': 'TOR',
        'Calgary Flames': 'CGY',
        'New York Islanders': 'NYI',
        'Colorado Avalanche': 'COL',
        'Tampa Bay Lightning': 'TBL',
        'Seattle Kraken': 'SEA',
        'Pittsburgh Penguins': 'PIT',
        'Los Angeles Kings': 'LAK',
        'Florida Panthers': 'FLA'
    }

    # Create a new column with abbreviations (to maybe join on the players table)
    TeamDF['team_abrev'] = TeamDF['teamFullName'].map(team_abbreviations)

    TeamDF.rename(columns={
        'gamesPlayed' : 'games_played',
        'goalsAgainst' : 'goals_against',
        'goalsAgainstPerGame' : 'goals_against_per_game',
        'powerPlayPct' : 'powerplay_percentage'
    }, inplace=True)

    return TeamDF[['team_abrev', 'fetch_date', 'games_played', 'goals_against', 'goals_against_per_game', 'powerplay_percentage']]


def CleanScheduleDataSource(ScheduleDf: pd.DataFrame) -> pd.DataFrame:

    ScheduleDf['game_date'] = datetime.today().date()

    ScheduleDf['away_team'] = ScheduleDf['awayTeam'].apply(lambda x: x['abbrev'] if isinstance(x, dict) else None)
    ScheduleDf['home_team'] = ScheduleDf['homeTeam'].apply(lambda x: x['abbrev'] if isinstance(x, dict) else None)

    return ScheduleDf[['game_date', 'home_team', 'away_team']]
