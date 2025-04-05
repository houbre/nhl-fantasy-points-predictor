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

app = Flask(__name__)
db = NHLDTBManager(Password="HelloThere")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    search_query = request.args.get('search', '')
    
    # Query the predictions table for the latest date
    query = """
    WITH latest_date AS (
        SELECT MAX(prediction_date) as max_date
        FROM predictions
    )
    SELECT p.name, p.team, p.opponent, p.predicted_points, p.actual_points, p.prediction_date
    FROM predictions p
    JOIN latest_date ld ON p.prediction_date = ld.max_date
    """
    
    if search_query:
        query += f" WHERE LOWER(p.name) LIKE LOWER('%{search_query}%')"
    
    query += " ORDER BY p.predicted_points DESC"
    
    try:
        predictions_df = pd.read_sql_query(query, db.engine)
        logger.info(f"Query results: {predictions_df.head()}")
        logger.info(f"Data types: {predictions_df.dtypes}")
        
        # Convert prediction_date to datetime if it's not already
        if 'prediction_date' in predictions_df.columns:
            # Check if the column is already datetime
            if not pd.api.types.is_datetime64_any_dtype(predictions_df['prediction_date']):
                # Try to convert to datetime
                try:
                    predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
                except Exception as e:
                    logger.error(f"Error converting prediction_date to datetime: {str(e)}")
                    # If conversion fails, use a default date
                    predictions_df['prediction_date'] = pd.Timestamp.now().normalize()
            
            # Format the date as string
            predictions_df['prediction_date'] = predictions_df['prediction_date'].dt.strftime('%Y-%m-%d')
        else:
            # If prediction_date column doesn't exist, add current date
            predictions_df['prediction_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        predictions = predictions_df.to_dict('records')
        logger.info(f"First prediction: {predictions[0] if predictions else 'No predictions found'}")
        return jsonify({'success': True, 'data': predictions})
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 