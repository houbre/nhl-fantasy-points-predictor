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
                        'timeOnIcePerGame_x']]
    
    CopyDF.rename(columns={
        'playerId': 'player_id',
        'skaterFullName_x': 'player_name',
        'gamesPlayed_x': 'games_played',
        'plusMinus': 'plus_minus',
        'ppPoints': 'pp_points',
        'blockedShots': 'blocked_shots'
    }, inplace=True)

    return CopyDF[['fetch_date', 'player_id', 'player_name', 'games_played', 'goals', 'assists', 'plus_minus', 'pp_points', 'shots', 'hits', 'blocked_shots', 'pp1_status']]


def CleanTeamDataSource(TeamDF: pd.DataFrame) -> pd.DataFrame:
    return TeamDF[['teamId', 'teamFullName', 'TeamAbbrevs', 'goalsAgainstPerGame']]


def CleanScheduleDataSource(ScheduleDf: pd.DataFrame) -> pd.DataFrame:
    return ScheduleDf[['homeTeam', 'awayTeam']]
