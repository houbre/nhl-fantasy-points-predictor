import logging
import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import date, timedelta
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NHLAPIConfig:
    """Configuration for NHL API client."""
    base_skater_url: str = "https://api.nhle.com/stats/rest/en/skater"
    base_team_url: str = "https://api.nhle.com/stats/rest/en/team"
    season_id: str = "20242025"  # Current season
    game_type_id: int = 2  # Regular season games

class NHLAPIClient:
    """Client for interacting with the NHL API."""

    def __init__(self, config: NHLAPIConfig = NHLAPIConfig()):
        self.config = config
        self.session = requests.Session()

    def _make_request(self, url: str) -> Dict:
        """Make a request to the NHL API."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise

    def get_skaters_summary_stats(self) -> pd.DataFrame:
        """
        Get season-long summary statistics for all skaters.
        Includes: goals, assists, powerplay points, plus/minus, shots
        """
        url = (f"{self.config.base_skater_url}/summary"
               f"?limit=-1&cayenneExp=seasonId={self.config.season_id}"
               f"%20and%20gameTypeId={self.config.game_type_id}")
        try:
            data = self._make_request(url)
            df = pd.DataFrame(data['data'])
            logger.info(f"Retrieved summary stats for {len(df)} players")
            return df
        except Exception as e:
            logger.error(f"Error getting season summary stats: {str(e)}")
            raise

    def get_skaters_miscellaneous_stats(self) -> pd.DataFrame:
        """
        Get season-long miscellaneous statistics for all skaters.
        Includes: hits, blocked shots
        """
        url = (f"{self.config.base_skater_url}/realtime"
               f"?limit=-1&cayenneExp=seasonId={self.config.season_id}"
               f"%20and%20gameTypeId={self.config.game_type_id}")
        try:
            data = self._make_request(url)
            df = pd.DataFrame(data['data'])
            logger.info(f"Retrieved miscellaneous stats for {len(df)} players")
            return df
        except Exception as e:
            logger.error(f"Error getting season miscellaneous stats: {str(e)}")
            raise

    def get_team_stats(self) -> pd.DataFrame:
        """
        Get season-long team statistics.
        Includes: goals against, games played
        """
        url = (f"{self.config.base_team_url}/summary"
               f"?limit=-1&cayenneExp=seasonId={self.config.season_id}"
               f"%20and%20gameTypeId={self.config.game_type_id}")
        try:
            data = self._make_request(url)
            df = pd.DataFrame(data['data'])
            logger.info(f"Retrieved team stats for {len(df)} teams")
            return df
        except Exception as e:
            logger.error(f"Error getting team stats: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    client = NHLAPIClient()
    
    try:
        # Get all required statistics
        summary_stats = client.get_skaters_summary_stats()
        print(summary_stats)
        
        misc_stats = client.get_skaters_miscellaneous_stats()
        print(misc_stats)
        
        team_stats = client.get_team_stats()
        print(team_stats)

        
    except Exception as e:
        print(f"Error in example usage: {e}")