import psycopg2
from sqlalchemy import create_engine
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NHLDTBManager:
    """
    Database manager for postgres transactions.
    """

    def __init__(self, Password: str):
        self.DB_NAME = "FantasyPointPredictor",
        self.DB_USER = "postgres",
        self.DB_PASSWORD = "HelloThere",
        self.DB_HOST = "localhost",
        self.DB_PORT = "8080"
        self.engine = create_engine("postgresql+psycopg2://postgres:HelloThere@localhost:8080/FantasyPointPredictor")

    def UpdatePlayersTable(self, PlayersDF: pd.DataFrame):
        try:
            PlayersDF.to_sql("player_season_stats", self.engine, if_exists="append", index=False)
            logger.info("Succesfully loaded the players stats into the database")

        except Exception as e:
            logger.error(f"Error while updating the player_season_stats table")
            raise

    def UpdateDailyGamesTable(self, GamesDF: pd.DataFrame):
        try:
            GamesDF.to_sql("daily_games", self.engine, if_exists="append", index=False)
            logger.info("Succesfully loaded the daily games info into the database")

        except Exception as e:
            logger.error(f"Error while updating the daily_games table")
            raise

    def UpdateTeamsTable(self, TeamsDF: pd.DataFrame):
        try:
            TeamsDF.to_sql("team_season_stats", self.engine, if_exists="append", index=False)
            logger.info("Succesfully loaded the team stats into the database")

        except Exception as e:
            logger.error(f"Error while updating the team_season_stats table")
            raise

