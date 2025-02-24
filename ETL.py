import APIClient
import DailyFaceoffPP1Scraper
import TransformDataSources
import pandas as pd


def main():
    # Fetch players and teams stats using the NHL's api.
    Client = APIClient.NHLAPIClient()
    SkaterSummaryStats = Client.get_skaters_summary_stats()
    SkaterMiscStats = Client.get_skaters_miscellaneous_stats()
    TeamStats = Client.get_team_stats()
    
    # Get all the players currently on the powerplay 1 of their respective team.
    PowerPlayScraper = DailyFaceoffPP1Scraper.PowerPlayScraper()
    PowerPlayInfo = PowerPlayScraper.get_all_pp1_players()

    # Combine the players' statistics into a cleaned dataframe.
    PlayerFantasyStats = TransformDataSources.CombinePlayerDataSources(SkaterSummaryStats, SkaterMiscStats, PowerPlayInfo)
    TeamDefensiveStats = TransformDataSources.CleanTeamDataSource(TeamStats)

    # Load the players' and team's statistics into postgres
    # TODO Create function

if __name__ == '__main__':
    main()