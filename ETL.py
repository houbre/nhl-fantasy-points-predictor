import APIClient
import DailyFaceoffPP1Scraper
import TransformDataSources
import DTBManager
import ModelPrediction
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nhl_etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_data():
    """
    Collect and store the latest NHL data.
    """
    try:
        # Fetch players, teams stats and schedule using the NHL's api
        Client = APIClient.NHLAPIClient()
        SkaterSummaryStats = Client.get_skaters_summary_stats()
        SkaterMiscStats = Client.get_skaters_miscellaneous_stats()
        TeamStats = Client.get_team_stats()
        TodaysSchedule = Client.get_todays_games_info()
        
        # Get all the players currently on the powerplay 1 of their respective team
        PowerPlayScraper = DailyFaceoffPP1Scraper.PowerPlayScraper()
        PowerPlayInfo = PowerPlayScraper.get_all_pp1_players()

        # Combine the players' statistics into a cleaned dataframe
        PlayerFantasyStats = TransformDataSources.CombinePlayerDataSources(SkaterSummaryStats, SkaterMiscStats, PowerPlayInfo)
        TeamDefensiveStats = TransformDataSources.CleanTeamDataSource(TeamStats)
        GameSchedule = TransformDataSources.CleanScheduleDataSource(TodaysSchedule)

        # Load the players' and team's statistics into postgres
        NHLDatabaseMgr = DTBManager.NHLDTBManager("HelloThere")
        NHLDatabaseMgr.UpdatePlayersTable(PlayerFantasyStats)
        NHLDatabaseMgr.UpdateTeamsTable(TeamDefensiveStats)
        NHLDatabaseMgr.UpdateDailyGamesTable(GameSchedule)
        
        logger.info("Successfully collected and stored NHL data")
        return True
        
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return False

def update_predictions():
    """
    Update yesterday's actual points and make new predictions for today.
    """
    try:
        predictor = ModelPrediction.NHLPointsPredictor()
        
        # Update yesterday's actual points
        logger.info("Updating yesterday's actual points...")
        predictor.UpdateActualPoints()
        
        # Get historical data for training
        logger.info("Getting historical data for model training...")
        historical_data = predictor.GetHistoricalData(daysBack=100)
        
        if historical_data.empty:
            logger.error("No historical data available for training")
            return False
            
        # Make predictions for today
        logger.info("Generating predictions for today's games...")
        predictions = predictor.PredictPoints(historical_data)
        
        if predictions.empty:
            logger.error("No predictions generated")
            return False
            
        # Save predictions to database
        logger.info("Saving predictions to database...")
        predictor.SavePredictions(predictions)
        
        # Save the trained model
        logger.info("Saving trained model...")
        model = predictor.TrainModel(historical_data)
        predictor.SaveModel(model, "nhl_points_predictor")
        
        logger.info("Successfully completed prediction workflow")
        return True
        
    except Exception as e:
        logger.error(f"Error in prediction workflow: {e}")
        return False

def main():
    """
    Main ETL pipeline that collects data and updates predictions.
    """
    start_time = datetime.now()
    logger.info("Starting NHL ETL pipeline")
    
    # Step 1: Collect and store data
    if not collect_data():
        logger.error("Data collection failed")
        return
        
    # Step 2: Update predictions
    if not update_predictions():
        logger.error("Prediction workflow failed")
        return
        
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"ETL pipeline completed successfully in {duration}")

if __name__ == '__main__':
    main()