import psycopg2
from sqlalchemy import create_engine
import pandas as pd


def main():
    data = {
    'team': ['Colorado'],
    'fetch_date': ['2025-03-05'],
    'games_played': [5],
    'goals_against': [10],
    'goals_against_per_game': [2]
    }

    df = pd.DataFrame(data)
    engine = create_engine("postgresql+psycopg2://postgres:HelloThere@localhost:8080/FantasyPointPredictor")
    df.to_sql("team_season_stats", engine, if_exists="append", index=False)

if __name__ == '__main__':
    main()