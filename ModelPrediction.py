import logging
import psycopg2
import pandas as pd
from datetime import timedelta, datetime


# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nhl_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NHLPointsPredictor:
    def __init__(self):
        '''
        Database connection parameters
        '''

        self.DB_HOST = "localhost",
        self.DB_NAME = "FantasyPointPredictor",
        self.DB_USER = "postgres",
        self.DB_PORT = "8080",
        self.DB_PASSOWRD = "HelloThere"

        # Fanatsy points system
        self.POINTS_GOAL = 6
        self.POINTS_ASSIST = 4
        self.POINTS_PLUS_MINUS = 1
        self.POINTS_PP = 2
        self.POINTS_SHOT = 0.9
        self.POINTS_HIT = 0.5
        self.POINTS_BLOCK = 1

    def GetConnection(self):
        """
        Establish and return a connection to the postgres database
        """

        try:
            conn = psycopg2.connect(
                host = self.DB_HOST,
                databaseName = self.DB_NAME,
                user = self.DB_USER,
                password = self.DB_PASSOWRD,
                port = self.DB_PORT
            )
            return conn
        
        except Exception as e:
            logger.error(f"Couldn't establish database connection: {e}")
            raise

    def GetHistoricalData(self, daysBack=100) -> pd.DataFrame:
        """
        Get historical data for model training. Only for the players on the teams that play today.
        This dataset contains per game stats averages, opponent teams stats, power play information and
        actual points these player made on previous predictions. 

        .P daysBack
        Number of days to look back for historical data.

        .R
        Dataframe containing preprocessed feature and target variables.
        """

        start_date = (datetime.now() - timedelta(days=daysBack)).strftime("%Y-%m-%d")

        conn = self.GetConnection()

        try:
            query=f"""
            WITH player_games AS (
                SELECT 
                    p.player_id,
                    p.player_full_name,
                    p.team_abrev,
                    g.game_date,
                    CASE 
                        WHEN p.team_abrev = g.home_team THEN g.away_team
                        ELSE g.home_team
                    END AS opponent_team,
                    p.games_played,
                    p.goals,
                    p.assists,
                    p.plus_minus,
                    p.pp_points,
                    p.shots,
                    p.hits,
                    p.blocked_shots,
                    p.pp1_status,
                    p.fetch_date
                FROM 
                    player_season_stats p
                JOIN 
                    daily_games g ON (p.team_abrev = g.home_team OR p.team_abrev = g.away_team)
                    AND p.fetch_date <= g.game_date 
                    AND p.fetch_date = (
                        SELECT MAX(fetch_date) 
                        FROM player_season_stats 
                        WHERE player_id = p.player_id AND fetch_date <= g.game_date
                    )
                WHERE 
                    g.game_date >= {start_date}
            )

            SELECT 
                pg.player_id,
                pg.player_full_name,
                pg.team_abrev,
                pg.opponent_team,
                pg.game_date,
                cast(pg.goals as DECIMAL)/cast(pg.games_played as DECIMAL) as goals_pg,
                cast(pg.assists as DECIMAL)/cast(pg.games_played as DECIMAL) as assists_pg,
                cast(pg.plus_minus as DECIMAL)/cast(pg.games_played as DECIMAL) as plus_minus_pg,
                cast(pg.pp_points as DECIMAL)/cast(pg.games_played as DECIMAL) as pp_points_pg,
                cast(pg.shots as DECIMAL)/cast(pg.games_played as DECIMAL) as shots_pg,
                cast(pg.hits as DECIMAL)/cast(pg.games_played as DECIMAL) as hits_pg,
                cast(pg.blocked_shots as DECIMAL)/cast(pg.games_played as DECIMAL) as blocked_shots_pg,
                pg.pp1_status,
                t_team.powerplay_percentage AS team_pp_percentage,
                t_opp.goals_against_per_game AS opponent_goals_against_per_game,
                COALESCE(pred.actual_points, NULL) AS actual_points
            FROM 
                player_games pg
            LEFT JOIN 
                team_season_stats t_team ON pg.team_abrev = t_team.team_abrev
                AND t_team.fetch_date = (
                    SELECT MAX(fetch_date) 
                    FROM team_season_stats 
                    WHERE team_abrev = pg.team_abrev AND fetch_date <= pg.game_date
                )
            LEFT JOIN 
                team_season_stats t_opp ON pg.opponent_team = t_opp.team_abrev
                AND t_opp.fetch_date = (
                    SELECT MAX(fetch_date) 
                    FROM team_season_stats 
                    WHERE team_abrev = pg.opponent_team AND fetch_date <= pg.game_date
                )
            LEFT JOIN
                predictions pred ON pg.player_id = pred.player_id AND pg.game_date = pred.prediction_date
            WHERE
                pg.games_played > 0
            --	and player_full_name = 'Adam Fantilli'
            ORDER BY
                pg.game_date DESC
            """

            df = pd.read_sql(query, conn)
            logger.info(f"Retrieved {len(df)} historical data points")
            
            return df

        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            raise

        finally:
            conn.close()





