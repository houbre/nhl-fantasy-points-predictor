import pandas as pd


def CombinePlayerDataSources(SummaryDF: pd.DataFrame, MiscellaneousDF: pd.DataFrame, PP1DF: pd.DataFrame) -> pd.DataFrame:

    MergedDF = SummaryDF.merge(MiscellaneousDF, on='playerId')

    pd.set_option('display.max_columns', None)  
    print(MergedDF)

    MergedDF['IsOnPP1'] = MergedDF['skaterFullName_x'].isin(PP1DF['player_name'])

    return MergedDF[['playerId', 
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
                     'IsOnPP1',
                     'timeOnIcePerGame_x']]


def CleanTeamDataSource(TeamDF: pd.DataFrame) -> pd.DataFrame:
    return TeamDF['teamId', 'teamFullName', 'TeamAbbrevs', 'goalsAgainstPerGame']
