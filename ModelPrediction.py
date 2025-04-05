import logging
import psycopg2
import pandas as pd
from datetime import timedelta, datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from xgboost import callback


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
        Database connection parameters and model configuration
        '''
        # Database connection parameters
        self.DB_HOST = "localhost"
        self.DB_NAME = "FantasyPointPredictor"
        self.DB_USER = "postgres"
        self.DB_PORT = 8080
        self.DB_PASSOWRD = "HelloThere"

        # Fantasy points system
        self.POINTS_GOAL = 6
        self.POINTS_ASSIST = 4
        self.POINTS_PLUS_MINUS = 1
        self.POINTS_PP = 2
        self.POINTS_SHOT = 0.9
        self.POINTS_HIT = 0.5
        self.POINTS_BLOCK = 1

        # Model features
        self.FEATURES = [
            'goals_pg', 'assists_pg', 'plus_minus_pg', 'pp_points_pg',
            'shots_pg', 'hits_pg', 'blocked_shots_pg', 'pp1_status',
            'team_pp_percentage', 'opponent_goals_against_per_game',
            'avg_prediction_error', 'prediction_accuracy', 'last_prediction_error',
            'prediction_trend'
        ]

    def GetConnection(self):
        """
        Establish and return a connection to the postgres database
        """
        try:
            conn = psycopg2.connect(
                host = self.DB_HOST,
                dbname = self.DB_NAME,
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
                    g.game_date >= '{start_date}'
            ),
            prediction_stats AS (
                SELECT 
                    player_id,
                    prediction_date,
                    predicted_points,
                    actual_points,
                    ABS(predicted_points - COALESCE(actual_points, 0)) as prediction_error,
                    CASE 
                        WHEN actual_points IS NOT NULL 
                        THEN ABS(predicted_points - actual_points) <= 1
                        ELSE NULL
                    END as accurate_prediction
                FROM predictions
                WHERE prediction_date >= '{start_date}'
            ),
            player_prediction_metrics AS (
                SELECT 
                    player_id,
                    prediction_date,
                    AVG(prediction_error) OVER (
                        PARTITION BY player_id 
                        ORDER BY prediction_date 
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                    ) as avg_prediction_error,
                    AVG(CASE WHEN accurate_prediction THEN 1 ELSE 0 END) OVER (
                        PARTITION BY player_id 
                        ORDER BY prediction_date 
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                    ) as prediction_accuracy,
                    LAG(prediction_error) OVER (
                        PARTITION BY player_id 
                        ORDER BY prediction_date
                    ) as last_prediction_error,
                    AVG(prediction_error) OVER (
                        PARTITION BY player_id 
                        ORDER BY prediction_date 
                        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                    ) - 
                    AVG(prediction_error) OVER (
                        PARTITION BY player_id 
                        ORDER BY prediction_date 
                        ROWS BETWEEN 6 PRECEDING AND 3 PRECEDING
                    ) as prediction_trend
                FROM prediction_stats
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
                COALESCE(pred.actual_points, NULL) AS actual_points,
                COALESCE(ppm.avg_prediction_error, 0) as avg_prediction_error,
                COALESCE(ppm.prediction_accuracy, 0) as prediction_accuracy,
                COALESCE(ppm.last_prediction_error, 0) as last_prediction_error,
                COALESCE(ppm.prediction_trend, 0) as prediction_trend
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
            LEFT JOIN
                player_prediction_metrics ppm ON pg.player_id = ppm.player_id AND pg.game_date = ppm.prediction_date
            WHERE
                pg.games_played > 0
            ORDER BY
                pg.game_date DESC
            """

            df = pd.read_sql(query, conn)
            logger.info(f"Retrieved {len(df)} historical data points")
            
            # Convert boolean pp1_status to int for model
            df['pp1_status'] = df['pp1_status'].astype(int)
            
            # Fill missing values with mean for numeric columns
            df[self.FEATURES] = df[self.FEATURES].fillna(df[self.FEATURES].mean())
            
            return df

        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            raise

        finally:
            conn.close()

    def TrainModel(self, df: pd.DataFrame) -> xgb.XGBRegressor:
        """
        Train the model on the historical data.
        
        Args:
            df: DataFrame containing historical data with features and target
            
        Returns:
            Trained XGBoost model
        """
        try:
            # Prepare features and target
            X = df[self.FEATURES]
            y = df['actual_points'].fillna(0)  # Fill missing actual points with 0
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train the model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            logger.info(f"Model trained successfully. MSE: {mse:.4f}, R2: {r2:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def PredictPoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the points for the players in the dataframe.
        
        Args:
            df: DataFrame containing player features for today's games
            
        Returns:
            DataFrame with predictions and player information
        """
        try:
            # Get today's date
            today = datetime.now().date()
            
            # Get today's games
            conn = self.GetConnection()
            query = """
            SELECT DISTINCT home_team, away_team
            FROM daily_games
            WHERE game_date = %s
            """
            
            today_games = pd.read_sql(query, conn, params=[today])
            
            if today_games.empty:
                logger.warning("No games found for today")
                return pd.DataFrame()
            
            # Get players from teams playing today
            teams_playing = list(set(today_games['home_team'].tolist() + today_games['away_team'].tolist()))
            teams_condition = " OR ".join([f"team_abrev = '{team}'" for team in teams_playing])
            
            # Get player stats and prediction metrics
            query = f"""
            WITH latest_stats AS (
                SELECT 
                    p.*,
                    ROW_NUMBER() OVER (PARTITION BY p.player_id ORDER BY p.fetch_date DESC) as rn
                FROM player_season_stats p
                WHERE {teams_condition}
            ),
            prediction_errors AS (
                SELECT 
                    player_id,
                    prediction_date,
                    ABS(predicted_points - COALESCE(actual_points, 0)) as prediction_error,
                    CASE WHEN ABS(predicted_points - COALESCE(actual_points, 0)) <= 1 THEN 1 ELSE 0 END as accurate_prediction
                FROM predictions
                WHERE player_id IN (SELECT player_id FROM latest_stats WHERE rn = 1)
            ),
            prediction_metrics AS (
                SELECT 
                    player_id,
                    AVG(prediction_error) as avg_prediction_error,
                    AVG(accurate_prediction) as prediction_accuracy,
                    MAX(CASE WHEN rn = 1 THEN prediction_error ELSE NULL END) as last_prediction_error,
                    (
                        AVG(CASE WHEN rn <= 3 THEN prediction_error ELSE NULL END) - 
                        AVG(CASE WHEN rn > 3 AND rn <= 6 THEN prediction_error ELSE NULL END)
                    ) as prediction_trend
                FROM (
                    SELECT 
                        player_id,
                        prediction_error,
                        accurate_prediction,
                        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY prediction_date DESC) as rn
                    FROM prediction_errors
                ) ranked_errors
                GROUP BY player_id
            )
            SELECT 
                ls.player_id,
                ls.player_full_name,
                ls.team_abrev,
                ls.games_played,
                ls.goals,
                ls.assists,
                ls.plus_minus,
                ls.pp_points,
                ls.shots,
                ls.hits,
                ls.blocked_shots,
                ls.pp1_status,
                COALESCE(pm.avg_prediction_error, 0) as avg_prediction_error,
                COALESCE(pm.prediction_accuracy, 0) as prediction_accuracy,
                COALESCE(pm.last_prediction_error, 0) as last_prediction_error,
                COALESCE(pm.prediction_trend, 0) as prediction_trend
            FROM latest_stats ls
            LEFT JOIN prediction_metrics pm ON ls.player_id = pm.player_id
            WHERE ls.rn = 1 AND ls.games_played > 0
            """
            
            players_df = pd.read_sql(query, conn)
            
            if players_df.empty:
                logger.warning("No players found for today's games")
                return pd.DataFrame()
            
            # Calculate per game averages
            players_df['goals_pg'] = players_df['goals'] / players_df['games_played']
            players_df['assists_pg'] = players_df['assists'] / players_df['games_played']
            players_df['plus_minus_pg'] = players_df['plus_minus'] / players_df['games_played']
            players_df['pp_points_pg'] = players_df['pp_points'] / players_df['games_played']
            players_df['shots_pg'] = players_df['shots'] / players_df['games_played']
            players_df['hits_pg'] = players_df['hits'] / players_df['games_played']
            players_df['blocked_shots_pg'] = players_df['blocked_shots'] / players_df['games_played']
            
            # Get opponent information
            players_df['opponent_team'] = players_df.apply(
                lambda row: today_games[today_games['home_team'] == row['team_abrev']]['away_team'].iloc[0]
                if row['team_abrev'] in today_games['home_team'].values
                else today_games[today_games['away_team'] == row['team_abrev']]['home_team'].iloc[0],
                axis=1
            )
            
            # Get team stats
            query = """
            WITH latest_team_stats AS (
                SELECT 
                    t.*,
                    ROW_NUMBER() OVER (PARTITION BY t.team_abrev ORDER BY t.fetch_date DESC) as rn
                FROM team_season_stats t
            )
            SELECT 
                team_abrev,
                powerplay_percentage as team_pp_percentage,
                goals_against_per_game
            FROM latest_team_stats
            WHERE rn = 1
            """
            
            team_stats = pd.read_sql(query, conn)
            
            # Merge team stats (for team's powerplay)
            players_df = players_df.merge(
                team_stats[['team_abrev', 'team_pp_percentage']],
                left_on='team_abrev',
                right_on='team_abrev'
            )
            
            # Merge team stats again (for opponent's goals against)
            players_df = players_df.merge(
                team_stats[['team_abrev', 'goals_against_per_game']],
                left_on='opponent_team',
                right_on='team_abrev',
                suffixes=('', '_opponent')
            ).rename(columns={'goals_against_per_game': 'opponent_goals_against_per_game'})
            
            # Prepare features for prediction
            X = players_df[self.FEATURES]
            
            # Train the model
            model = self.TrainModel(df)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'player_id': players_df['player_id'],
                'player_name': players_df['player_full_name'],
                'team': players_df['team_abrev'],
                'opponent': players_df['opponent_team'],
                'predicted_points': predictions
            })
            
            logger.info(f"Generated predictions for {len(results)} players")
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
            
        finally:
            conn.close()

    def SaveModel(self, model: object, model_name: str) -> None:
        """
        Save the model to the models folder.
        
        Args:
            model: Trained model to save
            model_name: Name of the model file
        """
        try:
            # Create models directory if it doesn't exist
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{models_dir}/{model_name}_{timestamp}.json"
            
            # Save the model
            model.save_model(filename)
            logger.info(f"Model saved successfully to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def UpdateActualPoints(self) -> None:
        """
        Update the actual points for yesterday's predictions based on the games played.
        Calculates points based on the difference between today's and yesterday's stats.
        """
        try:
            # Get yesterday's and today's dates
            today = (datetime.now()).date()
            yesterday = (datetime.now() - timedelta(days=1)).date()
            
            conn = self.GetConnection()
            
            # Get yesterday's games
            query = """
            SELECT DISTINCT home_team, away_team
            FROM daily_games
            WHERE game_date = %s
            """
            
            yesterday_games = pd.read_sql(query, conn, params=[yesterday])
            
            if yesterday_games.empty:
                logger.info("No games found for yesterday")
                return
            
            # Get players from teams that played yesterday
            teams_playing = list(set(yesterday_games['home_team'].tolist() + yesterday_games['away_team'].tolist()))
            teams_condition = " OR ".join([f"team_abrev = '{team}'" for team in teams_playing])
            
            # Get player stats for both today and yesterday
            query = f"""
            WITH player_stats AS (
                SELECT 
                    p.player_id,
                    p.player_full_name,
                    p.team_abrev,
                    CASE 
                        WHEN p.team_abrev = g.home_team THEN g.away_team
                        ELSE g.home_team
                    END AS opponent_team,
                    p.goals,
                    p.assists,
                    p.plus_minus,
                    p.pp_points,
                    p.shots,
                    p.hits,
                    p.blocked_shots,
                    p.fetch_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.player_id 
                        ORDER BY p.fetch_date DESC
                    ) as rn
                FROM 
                    player_season_stats p
                JOIN 
                    daily_games g ON (p.team_abrev = g.home_team OR p.team_abrev = g.away_team)
                WHERE 
                    g.game_date = %s
                    AND p.fetch_date <= %s
            )
            SELECT 
                ps1.player_id,
                ps1.player_full_name,
                ps1.team_abrev,
                ps1.opponent_team,
                ps1.goals - COALESCE(ps2.goals, 0) as goals,
                ps1.assists - COALESCE(ps2.assists, 0) as assists,
                ps1.plus_minus - COALESCE(ps2.plus_minus, 0) as plus_minus,
                ps1.pp_points - COALESCE(ps2.pp_points, 0) as pp_points,
                ps1.shots - COALESCE(ps2.shots, 0) as shots,
                ps1.hits - COALESCE(ps2.hits, 0) as hits,
                ps1.blocked_shots - COALESCE(ps2.blocked_shots, 0) as blocked_shots
            FROM 
                player_stats ps1
            LEFT JOIN 
                player_stats ps2 ON ps1.player_id = ps2.player_id AND ps2.rn = 2
            WHERE 
                ps1.rn = 1
            """
            
            players_stats = pd.read_sql(query, conn, params=[yesterday, today])
            
            if players_stats.empty:
                logger.warning("No player stats found for yesterday's games")
                return
            
            # Calculate actual fantasy points based on the difference in stats
            players_stats['actual_points'] = (
                players_stats['goals'] * self.POINTS_GOAL +
                players_stats['assists'] * self.POINTS_ASSIST +
                players_stats['plus_minus'] * self.POINTS_PLUS_MINUS +
                players_stats['pp_points'] * self.POINTS_PP +
                players_stats['shots'] * self.POINTS_SHOT +
                players_stats['hits'] * self.POINTS_HIT +
                players_stats['blocked_shots'] * self.POINTS_BLOCK
            )
            
            # Update predictions table with actual points
            for _, row in players_stats.iterrows():
                update_query = """
                UPDATE predictions 
                SET actual_points = %s
                WHERE player_id = %s 
                AND prediction_date = %s
                """
                conn.cursor().execute(update_query, (row['actual_points'], row['player_id'], yesterday))
            
            conn.commit()
            logger.info(f"Updated actual points for {len(players_stats)} players from yesterday's games")
            
        except Exception as e:
            logger.error(f"Error updating actual points: {e}")
            raise
            
        finally:
            conn.close()

    def SavePredictions(self, predictions_df: pd.DataFrame) -> None:
        """
        Save today's predictions to the predictions table.
        
        Args:
            predictions_df: DataFrame containing today's predictions
        """
        try:
            today = datetime.now().date()
            conn = self.GetConnection()
            
            # Generate model version using timestamp
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Insert predictions
            for _, row in predictions_df.iterrows():
                insert_query = """
                INSERT INTO predictions 
                (prediction_date, player_id, name, team, opponent, predicted_points, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                conn.cursor().execute(insert_query, (
                    today,
                    row['player_id'],
                    row['player_name'],
                    row['team'],
                    row['opponent'],
                    row['predicted_points'],
                    model_version
                ))
            
            conn.commit()
            logger.info(f"Saved {len(predictions_df)} predictions for today's games with model version {model_version}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise
            
        finally:
            conn.close()