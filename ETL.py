import APIClient
import DailyFaceoffPP1Scraper
import TransformDataSources
import DTBManager
import pandas as pd


def main():
    # Fetch players, teams stats and schedule using the NHL's api.
    Client = APIClient.NHLAPIClient()
    SkaterSummaryStats = Client.get_skaters_summary_stats()
    SkaterSummaryStats.to_csv('SkaterSummaryStats.csv')
    SkaterMiscStats = Client.get_skaters_miscellaneous_stats()
    TeamStats = Client.get_team_stats()
    TodaysSchedule = Client.get_todays_games_info()
    
    # Get all the players currently on the powerplay 1 of their respective team.
    PowerPlayScraper = DailyFaceoffPP1Scraper.PowerPlayScraper()
    PowerPlayInfo = PowerPlayScraper.get_all_pp1_players()

    # Combine the players' statistics into a cleaned dataframe.
    PlayerFantasyStats = TransformDataSources.CombinePlayerDataSources(SkaterSummaryStats, SkaterMiscStats, PowerPlayInfo)
    TeamDefensiveStats = TransformDataSources.CleanTeamDataSource(TeamStats)
    GameSchedule = TransformDataSources.CleanScheduleDataSource(TodaysSchedule)

    # Load the players' and team's statistics into postgres
    NHLDatabaseMgr = DTBManager.NHLDTBManager("HelloThere")
    NHLDatabaseMgr.UpdatePlayersTable(PlayerFantasyStats)
    NHLDatabaseMgr.UpdateTeamsTable(TeamDefensiveStats)
    NHLDatabaseMgr.UpdateDailyGamesTable(GameSchedule)

if __name__ == '__main__':
    main()