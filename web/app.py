from flask import Flask, render_template, request, jsonify
import sys
import os
import logging
from datetime import datetime

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

@app.route('/api/dates')
def get_dates():
    """Get all available prediction dates"""
    try:
        query = """
        SELECT DISTINCT prediction_date
        FROM predictions
        ORDER BY prediction_date DESC
        """
        
        dates_df = pd.read_sql_query(query, db.engine)
        
        # Convert prediction_date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(dates_df['prediction_date']):
            dates_df['prediction_date'] = pd.to_datetime(dates_df['prediction_date'])
        
        dates = dates_df['prediction_date'].dt.strftime('%Y-%m-%d').tolist()
        
        return jsonify({'success': True, 'data': dates})
    except Exception as e:
        logger.error(f"Error fetching dates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predictions')
def get_predictions():
    search_query = request.args.get('search', '')
    selected_date = request.args.get('date', '')
    
    # Query the predictions table for the selected date or latest date
    base_query = """
    SELECT p.name, p.team, p.opponent, p.predicted_points, p.actual_points, p.prediction_date
    FROM predictions p
    """
    
    # Add date filter if a specific date is selected
    if selected_date:
        base_query += " WHERE p.prediction_date = :selected_date"
    else:
        # If no date selected, get the latest date
        base_query = """
        WITH latest_date AS (
            SELECT MAX(prediction_date) as max_date
            FROM predictions
        )
        SELECT p.name, p.team, p.opponent, p.predicted_points, p.actual_points, p.prediction_date
        FROM predictions p
        JOIN latest_date ld ON p.prediction_date = ld.max_date
        """
    
    # Add search filter if needed
    if search_query:
        if selected_date:
            base_query += " AND LOWER(p.name) LIKE LOWER(:search_pattern)"
        else:
            base_query += " WHERE LOWER(p.name) LIKE LOWER(:search_pattern)"
        search_pattern = f"%{search_query}%"
        query = text(base_query).bindparams(search_pattern=search_pattern)
    else:
        query = text(base_query)
    
    # Bind the date parameter if a specific date is selected
    if selected_date:
        query = query.bindparams(selected_date=selected_date)
    
    # Always order by predicted_points in descending order
    base_query += " ORDER BY p.predicted_points DESC"
    
    try:
        logger.info(f"Executing query with search: {search_query}, date: {selected_date}")
        predictions_df = pd.read_sql_query(query, db.engine)
        logger.info(f"Query results: {predictions_df.head()}")
        logger.info(f"Data types: {predictions_df.dtypes}")
        
        # Convert prediction_date to datetime if it's not already
        if 'prediction_date' in predictions_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(predictions_df['prediction_date']):
                predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
            
            # Format the date as string
            predictions_df['prediction_date'] = predictions_df['prediction_date'].dt.strftime('%Y-%m-%d')
        
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