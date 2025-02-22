import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List
from dataclasses import dataclass
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TeamInfo:
    """NHL team information for URL mapping."""
    name: str
    url_name: str

class PowerPlayScraper:
    """Scraper for DailyFaceoff powerplay unit information."""
    
    def __init__(self):
        self.base_url = "https://www.dailyfaceoff.com/teams"
        # Headers to mimic a browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Define team mappings
        self.teams = [
            TeamInfo("Anaheim Ducks", "anaheim-ducks"),
            TeamInfo("Boston Bruins", "boston-bruins"),
            TeamInfo("Buffalo Sabres", "buffalo-sabres"),
            TeamInfo("Calgary Flames", "calgary-flames"),
            TeamInfo("Carolina Hurricanes", "carolina-hurricanes"),
            TeamInfo("Chicago Blackhawks", "chicago-blackhawks"),
            TeamInfo("Colorado Avalanche", "colorado-avalanche"),
            TeamInfo("Columbus Blue Jackets", "columbus-blue-jackets"),
            TeamInfo("Dallas Stars", "dallas-stars"),
            TeamInfo("Detroit Red Wings", "detroit-red-wings"),
            TeamInfo("Edmonton Oilers", "edmonton-oilers"),
            TeamInfo("Florida Panthers", "florida-panthers"),
            TeamInfo("Los Angeles Kings", "los-angeles-kings"),
            TeamInfo("Minnesota Wild", "minnesota-wild"),
            TeamInfo("Montreal Canadiens", "montreal-canadiens"),
            TeamInfo("Nashville Predators", "nashville-predators"),
            TeamInfo("New Jersey Devils", "new-jersey-devils"),
            TeamInfo("New York Islanders", "new-york-islanders"),
            TeamInfo("New York Rangers", "new-york-rangers"),
            TeamInfo("Ottawa Senators", "ottawa-senators"),
            TeamInfo("Philadelphia Flyers", "philadelphia-flyers"),
            TeamInfo("Pittsburgh Penguins", "pittsburgh-penguins"),
            TeamInfo("San Jose Sharks", "san-jose-sharks"),
            TeamInfo("Seattle Kraken", "seattle-kraken"),
            TeamInfo("St. Louis Blues", "st-louis-blues"),
            TeamInfo("Tampa Bay Lightning", "tampa-bay-lightning"),
            TeamInfo("Toronto Maple Leafs", "toronto-maple-leafs"),
            TeamInfo("Utah Hockey Club", "utah-hockey-club"),
            TeamInfo("Vancouver Canucks", "vancouver-canucks"),
            TeamInfo("Vegas Golden Knights", "vegas-golden-knights"),
            TeamInfo("Washington Capitals", "washington-capitals"),
            TeamInfo("Winnipeg Jets", "winnipeg-jets")
        ]

    def get_team_page(self, team: TeamInfo) -> str:
        """
        Fetch the team's lineup page from DailyFaceoff.
        """

        url = f"{self.base_url}/{team.url_name}/line-combinations"
        try:
            # Add a small random delay to be respectful to the server
            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {team.name} page: {str(e)}")
            return ""

    def extract_pp1_players(self, html_content: str) -> List[str]:
        """
        Extract players on PP1 from team page HTML using the specific div structure.
        """

        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
            
        # Find all span elements with the player name class
        player_spans = soup.find_all('span', class_="text-xs font-bold uppercase xl:text-base")

        # Extract and clean names
        player_names = [span.text.strip() for span in player_spans]

        # only return power play 1 names
        return player_names[18:23]

    def get_all_pp1_players(self) -> pd.DataFrame:
        """
        Get all players on PP1 units across the league.
        """

        all_players_data = []
        
        for team in self.teams:
            logger.info(f"Fetching PP1 data for {team.name}")
            html_content = self.get_team_page(team)
            pp1_players = self.extract_pp1_players(html_content)
            
            # Add players to the dataset
            for player in pp1_players:
                all_players_data.append({
                    'player_name': player,
                    'team_name': team.name,
                    'on_pp1': True
                })
            
            logger.info(f"Found {len(pp1_players)} players on PP1 for {team.name}")
        
        # Create DataFrame
        df = pd.DataFrame(all_players_data)
        
        # Add entry for players not on PP1
        # This will be useful when merging with the NHL API data
        df = df.merge(
            pd.DataFrame({'on_pp1': [False]}),
            how='outer',
            on='on_pp1'
        )
        
        logger.info(f"Total PP1 players found: {len(df[df['on_pp1']])}")
        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str = 'pp1_players.csv'):
        """
        Save the PP1 players data to a CSV file.
        """

        df.to_csv(filename, index=False)
        logger.info(f"Saved PP1 data to {filename}")

# Example usage
if __name__ == "__main__":
    scraper = PowerPlayScraper()
    
    try:
        # Get all PP1 players
        pp1_data = scraper.get_all_pp1_players()
        
        # Display sample of the data
        print("\nPP1 Players Sample:")
        print(pp1_data.head())
        
        # Save to CSV
        scraper.save_to_csv(pp1_data)
        
        # Print some statistics
        print(f"\nTotal PP1 players found: {len(pp1_data)}")
        print("\nPlayers per team:")
        print(pp1_data.groupby('team_name').size())
        
    except Exception as e:
        print(f"Error in example usage: {e}")