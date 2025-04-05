from flask import Flask, render_template, request, jsonify
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DTBManager import NHLDTBManager
import pandas as pd
from sqlalchemy import text

app = Flask(__name__)
db = NHLDTBManager(Password="HelloThere")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    search_query = request.args.get('search', '')
    
    # Query the predictions table for the latest date
    base_query = """
    WITH latest_date AS (
        SELECT MAX(prediction_date) as max_date
        FROM predictions
    )
    SELECT p.name, p.team, p.opponent, p.predicted_points, p.actual_points
    FROM predictions p
    JOIN latest_date ld ON p.prediction_date = ld.max_date
    """
    
    # Add WHERE clause for search if needed
    if search_query:
        base_query += " WHERE LOWER(p.name) LIKE LOWER(:search_pattern)"
        search_pattern = f"%{search_query}%"
        query = text(base_query).bindparams(search_pattern=search_pattern)
    else:
        query = text(base_query)
    
    # Always order by predicted_points in descending order
    base_query += " ORDER BY p.predicted_points DESC"
    
    try:
        logger.info(f"Executing query with search: {search_query}")
        predictions_df = pd.read_sql_query(query, db.engine)
        logger.info(f"Query results: {predictions_df.head()}")
        logger.info(f"Data types: {predictions_df.dtypes}")
        
        # Ensure the DataFrame is sorted by predicted_points in descending order
        predictions_df = predictions_df.sort_values(by='predicted_points', ascending=False)
        
        predictions = predictions_df.to_dict('records')
        logger.info(f"First prediction: {predictions[0] if predictions else 'No predictions found'}")
        return jsonify({'success': True, 'data': predictions})
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 