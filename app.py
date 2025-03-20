from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import google.generativeai as genai
import os
import tempfile
import json
from datetime import datetime, timedelta
import logging
from io import StringIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyBsCeTkekViD7qFma8TfWZSvfwrL0sUpmE"
genai.configure(api_key=GEMINI_API_KEY)

# Constants
ALLOWED_CATEGORIES = ['dining', 'shopping', 'groceries', 'entertainment', 'travel', 'utilities', 'misc']
CACHE_FILE = 'spending_forecast_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {CACHE_FILE} is corrupted. Creating a new one.")
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def safe_forecast(series, forecast_periods, seasonal_periods=6):
    """Safely forecast the series with error handling"""
    try:
        # Apply Holt-Winters exponential smoothing
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods
        )
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=forecast_periods)
        
        # Ensure no negative values
        forecast = np.maximum(forecast, 0)
        return forecast
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        # Fallback to a simple moving average if modeling fails
        mean_value = series.mean()
        return np.array([mean_value] * forecast_periods)

def parse_spending_data(file_path):
    """Parse spending data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Ensure required columns are present
        required_columns = ['date', 'amount', 'category']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Normalize categories
        df['category'] = df['category'].str.lower()
        
        # Group by date and category, sum the amounts
        df = df.groupby(['date', 'category'])['amount'].sum().reset_index()
        
        return df
    except Exception as e:
        logger.error(f"Error parsing spending data: {str(e)}")
        raise

def parse_income_data(file_path):
    """Parse income data from CSV file"""
    try:
        income_df = pd.read_csv(file_path)
        
        # Ensure required columns are present
        required_columns = ['date', 'amount']
        for col in required_columns:
            if col not in income_df.columns:
                raise ValueError(f"Missing required column in income data: {col}")
        
        # Convert date to datetime
        income_df['date'] = pd.to_datetime(income_df['date'])
        
        # Group by date, sum the amounts
        income_df = income_df.groupby('date')['amount'].sum().reset_index()
        
        return income_df
    except Exception as e:
        logger.error(f"Error parsing income data: {str(e)}")
        raise

def calculate_savings_rate(spending_df, income_df):
    """Calculate savings rate based on income and spending"""
    try:
        # Resample both dataframes to monthly
        spending_monthly = spending_df.set_index('date').groupby(pd.Grouper(freq='M'))['amount'].sum()
        income_monthly = income_df.set_index('date').groupby(pd.Grouper(freq='M'))['amount'].sum()
        
        # Align the indexes
        common_dates = spending_monthly.index.intersection(income_monthly.index)
        spending_aligned = spending_monthly.loc[common_dates]
        income_aligned = income_monthly.loc[common_dates]
        
        # Calculate savings (income - spending)
        savings = income_aligned - spending_aligned
        
        # Calculate savings rate (savings / income)
        savings_rate = (savings / income_aligned) * 100
        
        return {
            'dates': common_dates.strftime('%Y-%m-%d').tolist(),
            'income': income_aligned.values.tolist(),
            'spending': spending_aligned.values.tolist(),
            'savings': savings.values.tolist(),
            'savings_rate': savings_rate.values.tolist()
        }
    except Exception as e:
        logger.error(f"Error calculating savings rate: {str(e)}")
        return None

def forecast_income(income_df, forecast_periods=12):
    """Generate forecast for income"""
    try:
        # Resample data to monthly frequency
        income_monthly = income_df.set_index('date').groupby(pd.Grouper(freq='M'))['amount'].sum()
        
        # Ensure we have enough data
        if len(income_monthly) < 3:
            logger.warning("Not enough income data to generate forecast")
            return None
        
        # Generate forecast
        forecast_values = safe_forecast(income_monthly, forecast_periods)
        
        # Create forecast dates
        last_date = income_monthly.index.max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                      periods=forecast_periods, 
                                      freq='M')
        
        return {
            'historical': {
                'dates': income_monthly.index.strftime('%Y-%m-%d').tolist(),
                'values': income_monthly.values.tolist()
            },
            'forecast': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'values': forecast_values.tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error forecasting income: {str(e)}")
        return None

def generate_forecast(spending_df, income_df=None, forecast_periods=12):
    """Generate forecasts for each spending category and income"""
    
    # Set date as index
    spending_df['date'] = pd.to_datetime(spending_df['date'])
    
    # Get the latest date in the dataset
    latest_date = spending_df['date'].max()
    
    # Resample data to monthly frequency
    df_monthly = spending_df.set_index('date').groupby([pd.Grouper(freq='M'), 'category'])['amount'].sum().reset_index()
    
    results = {
        'historical': {},
        'forecast': {}
    }
    
    # Process each category
    for category in df_monthly['category'].unique():
        cat_data = df_monthly[df_monthly['category'] == category]
        
        # Ensure we have enough data
        if len(cat_data) < 3:
            logger.warning(f"Not enough data for category {category} to generate forecast")
            continue
        
        # Set up time series
        cat_series = cat_data.set_index('date')['amount']
        
        # Generate forecast
        forecast_values = safe_forecast(cat_series, forecast_periods)
        
        # Create forecast dates
        last_date = cat_series.index.max()
        if pd.isna(last_date):
            last_date = latest_date
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                      periods=forecast_periods, 
                                      freq='M')
        
        # Store historical data
        results['historical'][category] = {
            'dates': cat_series.index.strftime('%Y-%m-%d').tolist(),
            'values': cat_series.values.tolist()
        }
        
        # Store forecast data
        results['forecast'][category] = {
            'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'values': forecast_values.tolist()
        }
    
    # Calculate total spending (historical and forecast)
    if df_monthly.empty:
        logger.warning("No spending data available for total calculation")
    else:
        # Group by date only to get total spending
        total_spending = df_monthly.groupby('date')['amount'].sum()
        
        # Forecast total spending
        if len(total_spending) >= 3:
            total_forecast = safe_forecast(total_spending, forecast_periods)
            
            # Store total spending data
            results['historical']['total'] = {
                'dates': total_spending.index.strftime('%Y-%m-%d').tolist(),
                'values': total_spending.values.tolist()
            }
            
            results['forecast']['total'] = {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'values': total_forecast.tolist()
            }
    
    # Add income forecast if income data is provided
    if income_df is not None:
        income_forecast = forecast_income(income_df, forecast_periods)
        if income_forecast:
            results['historical']['income'] = income_forecast['historical']
            results['forecast']['income'] = income_forecast['forecast']
            
            # Calculate savings forecast
            if 'total' in results['forecast']:
                savings_forecast = []
                for i in range(len(results['forecast']['income']['values'])):
                    savings_forecast.append(
                        results['forecast']['income']['values'][i] - results['forecast']['total']['values'][i]
                    )
                
                results['forecast']['savings'] = {
                    'dates': results['forecast']['income']['dates'],
                    'values': savings_forecast
                }
    
    # Add savings rate if both income and spending are available
    if income_df is not None and not spending_df.empty:
        savings_data = calculate_savings_rate(spending_df, income_df)
        if savings_data:
            results['savings_analysis'] = savings_data
    
    return results

def generate_insights(forecast_data, user_query=""):
    """Generate insights based on forecast data using Gemini AI"""
    
    # Prepare the data for Gemini
    categories = list(forecast_data['historical'].keys())
    
    # Build the prompt
    prompt = f"""You are a financial analyst specializing in personal finance. Analyze the following spending data across categories: {', '.join(categories)}.

Historical and forecast data:
"""
    
    # Add details for each category
    for category in categories:
        if category == 'income':
            prompt += f"\nIncome:\n"
        elif category == 'total':
            prompt += f"\nTotal Spending:\n"
        elif category == 'savings':
            prompt += f"\nSavings (Income - Total Spending):\n"
        else:
            prompt += f"\n{category.capitalize()} spending:\n"
            
        # Add historical data if available
        if category in forecast_data['historical']:
            prompt += f"  Historical: {', '.join([f'({date}, ${value:.2f})' for date, value in zip(forecast_data['historical'][category]['dates'], forecast_data['historical'][category]['values'])])}\n"
        
        # Add forecast data if available
        if category in forecast_data['forecast']:
            prompt += f"  Forecast: {', '.join([f'({date}, ${value:.2f})' for date, value in zip(forecast_data['forecast'][category]['dates'], forecast_data['forecast'][category]['values'])])}\n"
    
    # Add savings rate analysis if available
    if 'savings_analysis' in forecast_data:
        prompt += "\nSavings Rate Analysis:\n"
        for i, (date, income, spending, savings, rate) in enumerate(zip(
            forecast_data['savings_analysis']['dates'],
            forecast_data['savings_analysis']['income'],
            forecast_data['savings_analysis']['spending'],
            forecast_data['savings_analysis']['savings'],
            forecast_data['savings_analysis']['savings_rate']
        )):
            prompt += f"  {date}: Income=${income:.2f}, Spending=${spending:.2f}, Savings=${savings:.2f}, Rate={rate:.2f}%\n"
    
    # Add analysis instructions
    prompt += """
Provide a comprehensive analysis including:
1. Overall spending trends and patterns
2. Category-specific insights with any notable increases or decreases
3. Seasonal patterns or anomalies in the data
4. Income vs. spending analysis and savings trends
5. Recommendations for budget optimization based on category-specific spending
6. A clear forecast summary for the next period
7. Potential areas of concern or saving opportunities

Consider relevant biases such as:
- Seasonal spending patterns
- Inflation effects on specific categories
- Income stability and growth trends
- Relationship between discretionary and non-discretionary spending

Be practical and explain like a financial expert. Provide actionable recommendations.
"""

    # Add user query if provided
    if user_query:
        prompt += f"\nAdditionally, address this specific question from the user: {user_query}"
    
    try:
        # Request insights from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Process and return the insights
        insights = response.text
        return insights
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return "Unable to generate insights due to an error. Please try again later."

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/forecast', methods=['POST'])
def create_forecast():
    """Create a forecast based on uploaded CSV data"""
    try:
        # Check if spending CSV file is uploaded
        if 'spending_file' not in request.files:
            return jsonify({'error': 'No spending file uploaded'}), 400
        
        spending_file = request.files['spending_file']
        if spending_file.filename == '':
            return jsonify({'error': 'No spending file selected'}), 400
        
        # Get forecast periods from request
        forecast_periods = int(request.form.get('forecast_periods', 12))
        
        # Get optional user query
        user_query = request.form.get('user_query', '')
        
        # Save the uploaded spending file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_spending_file:
            spending_file.save(temp_spending_file.name)
            temp_spending_path = temp_spending_file.name
        
        # Check if income file is uploaded
        income_df = None
        temp_income_path = None
        if 'income_file' in request.files and request.files['income_file'].filename != '':
            income_file = request.files['income_file']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_income_file:
                income_file.save(temp_income_file.name)
                temp_income_path = temp_income_file.name
                
            # Parse income data
            try:
                income_df = parse_income_data(temp_income_path)
            except Exception as e:
                logger.error(f"Error parsing income data: {str(e)}")
                return jsonify({'error': f"Income data parsing error: {str(e)}"}), 400
        
        try:
            # Parse the spending CSV file
            spending_df = parse_spending_data(temp_spending_path)
            
            # Generate forecast with both spending and income data if available
            forecast_data = generate_forecast(spending_df, income_df, forecast_periods)
            
            # Generate insights
            insights = generate_insights(forecast_data, user_query)
            
            # Create response
            response = {
                'status': 'success',
                'data': forecast_data,
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_spending_path):
                os.unlink(temp_spending_path)
            if temp_income_path and os.path.exists(temp_income_path):
                os.unlink(temp_income_path)
    
    except Exception as e:
        logger.error(f"Error in forecast creation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/json-forecast', methods=['POST'])
def create_json_forecast():
    """Create a forecast based on JSON data instead of CSV files"""
    try:
        print("Received request at /api/json-forecast")
        
        # Get JSON data from request
        data = request.json
        if not data:
            print("Error: No data provided")
            return jsonify({'error': 'No data provided'}), 400
        
        print("Received JSON data:", data)
        
        # Extract spending data
        if 'spending' not in data:
            print("Error: Missing 'spending' key in request")
            return jsonify({'error': 'Spending data is required'}), 400
        
        spending_data = data['spending']
        print("Raw spending data string:\n", spending_data)
        
        # Create a DataFrame from the spending data
        try:
            spending_df = pd.read_csv(StringIO(spending_data))
            print("Parsed spending DataFrame:\n", spending_df.head())
            
            # Ensure required columns
            required_columns = ['date', 'category', 'amount']
            for col in required_columns:
                if col not in spending_df.columns:
                    print(f"Error: Missing required column in spending data: {col}")
                    raise ValueError(f"Missing required column in spending data: {col}")
            
            # Convert date column to datetime
            spending_df['date'] = pd.to_datetime(spending_df['date'])
            print("Spending DataFrame after processing:\n", spending_df.head())
        except Exception as e:
            print(f"Error processing spending data: {e}")
            return jsonify({'error': f"Invalid spending data format: {str(e)}"}), 400
        
        # Extract income data if provided
        income_df = None
        if 'income' in data and data['income']:
            try:
                print("Raw income data string:\n", data['income'])
                
                income_df = pd.read_csv(StringIO(data['income']))
                print("Parsed income DataFrame:\n", income_df.head())

                # Ensure required columns
                required_columns = ['date', 'amount']
                for col in required_columns:
                    if col not in income_df.columns:
                        print(f"Error: Missing required column in income data: {col}")
                        raise ValueError(f"Missing required column in income data: {col}")
                
                # Convert date column to datetime
                income_df['date'] = pd.to_datetime(income_df['date'])
                print("Income DataFrame after processing:\n", income_df.head())
            except Exception as e:
                print(f"Error processing income data: {e}")
                return jsonify({'error': f"Invalid income data format: {str(e)}"}), 400
        
        # Get forecast periods from request
        forecast_periods = int(data.get('forecast_periods', 12))
        print(f"Forecast periods: {forecast_periods}")

        # Get optional user query
        user_query = data.get('user_query', '')
        print(f"User query: {user_query}")

        # Generate forecast (Placeholder function)
        forecast_data = generate_forecast(spending_df, income_df, forecast_periods)
        
        # Generate insights (Placeholder function)
        insights = generate_insights(forecast_data, user_query)

        # Create response
        response = {
            'status': 'success',
            'data': forecast_data,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }

        print("Final response:\n", response)
        return jsonify(response)
    
    except Exception as e:
        print(f"Unhandled error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Generate sample spending and income data for testing"""
    try:
        # Generate sample data
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate spending data
        spending_data = []
        for date in dates:
            for category in ALLOWED_CATEGORIES:
                # Generate random amounts with some patterns
                base_amount = {
                    'dining': 30,
                    'shopping': 50,
                    'groceries': 80,
                    'entertainment': 40,
                    'travel': 100,
                    'utilities': 120,
                    'misc': 20
                }[category]
                
                # Add some seasonality and randomness
                seasonality = 1.0
                if category == 'shopping' and (date.month == 11 or date.month == 12):
                    seasonality = 1.5  # More shopping during holidays
                
                if category == 'dining' and (date.weekday() >= 5):  # Weekend
                    seasonality = 1.3  # More dining on weekends
                    
                if category == 'travel' and (date.month in [6, 7, 8]):
                    seasonality = 1.8  # More travel during summer
                
                if category == 'utilities' and (date.month in [1, 2, 7, 8]):
                    seasonality = 1.2  # Higher utilities in winter and summer
                
                # Some days might not have spending in this category
                if np.random.random() < 0.7:  # 70% chance of spending in this category on this day
                    amount = base_amount * seasonality * np.random.uniform(0.7, 1.3)
                    
                    spending_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'category': category,
                        'amount': round(amount, 2)
                    })
        
        # Generate income data (typically twice a month)
        income_data = []
        for date in pd.date_range(start=start_date, end=end_date, freq='SM'):  # Semi-month frequency
            base_income = 3000  # Base monthly income
            
            # Add some raises and bonuses over time
            time_factor = (date - start_date).days / 365  # Fraction of year passed
            income_growth = 1.0 + (time_factor * 0.05)  # 5% annual increase
            
            # Add bonuses in December and mid-year
            bonus = 0
            if date.month == 12:
                bonus = base_income * 0.2  # 20% holiday bonus
            elif date.month == 6:
                bonus = base_income * 0.1  # 10% mid-year bonus
            
            # Add some randomness
            income = (base_income * income_growth + bonus) * np.random.uniform(0.95, 1.05)
            
            income_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'amount': round(income, 2)
            })
        
        # Create DataFrames
        spending_df = pd.DataFrame(spending_data)
        income_df = pd.DataFrame(income_data)
        
        # Save to temporary CSV files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_spending_file:
            spending_df.to_csv(temp_spending_file.name, index=False)
            temp_spending_path = temp_spending_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_income_file:
            income_df.to_csv(temp_income_file.name, index=False)
            temp_income_path = temp_income_file.name
        
        # Return the URLs to download the sample data
        return jsonify({
            'status': 'success',
            'sample_spending_url': f'/api/download-sample?file=spending_{os.path.basename(temp_spending_path)}',
            'sample_income_url': f'/api/download-sample?file=income_{os.path.basename(temp_income_path)}',
            'sample_data_expiry': (datetime.now() + timedelta(minutes=5)).isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-sample', methods=['GET'])
def download_sample():
    """Download the generated sample data"""
    try:
        file_name = request.args.get('file', '')
        if not file_name:
            return jsonify({'error': 'No file specified'}), 400
        
        file_path = os.path.join(tempfile.gettempdir(), file_name.split('_')[-1])
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Sample file not found or expired'}), 404
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Clean up the file
        os.unlink(file_path)
        
        return jsonify({'status': 'success', 'data': content})
    
    except Exception as e:
        logger.error(f"Error downloading sample data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get the list of allowed spending categories"""
    return jsonify({
        'status': 'success',
        'categories': ALLOWED_CATEGORIES
    })

@app.route('/api/documentation', methods=['GET'])
def get_api_docs():
    """Get API documentation"""
    docs = {
        'api_version': '1.0.0',
        'endpoints': [
            {
                'path': '/api/health',
                'method': 'GET',
                'description': 'Health check endpoint to verify API is operational',
                'parameters': [],
                'response': {'status': 'healthy', 'timestamp': 'ISO date string', 'version': 'API version'}
            },
            {
                'path': '/api/forecast',
                'method': 'POST',
                'description': 'Generate spending forecast from CSV files',
                'parameters': [
                    {'name': 'spending_file', 'type': 'file', 'required': True, 'description': 'CSV file with spending data (columns: date, amount, category)'},
                    {'name': 'income_file', 'type': 'file', 'required': False, 'description': 'CSV file with income data (columns: date, amount)'},
                    {'name': 'forecast_periods', 'type': 'integer', 'required': False, 'default': 12, 'description': 'Number of periods to forecast'},
                    {'name': 'user_query', 'type': 'string', 'required': False, 'description': 'User-specific analysis question'}
                ],
                'response': {'status': 'success', 'data': {}, 'insights': 'string', 'timestamp': 'ISO date string'}
            },
            {
                'path': '/api/json-forecast',
                'method': 'POST',
                'description': 'Generate spending forecast from JSON data',
                'parameters': [
                    {'name': 'spending', 'type': 'array', 'required': True, 'description': 'Array of spending objects with date, amount, and category'},
                    {'name': 'income', 'type': 'array', 'required': False, 'description': 'Array of income objects with date and amount'},
                    {'name': 'forecast_periods', 'type': 'integer', 'required': False, 'default': 12, 'description': 'Number of periods to forecast'},
                    {'name': 'user_query', 'type': 'string', 'required': False, 'description': 'User-specific analysis question'}
                ],
                'response': {'status': 'success', 'data': {}, 'insights': 'string', 'timestamp': 'ISO date string'}
            },
            {
                'path': '/api/sample-data',
                'method': 'GET',
                'description': 'Generate sample data for testing',
                'parameters': [],
                'response': {'status': 'success', 'sample_spending_url': 'string', 'sample_income_url': 'string', 'sample_data_expiry': 'ISO date string'}
            },
            {
                'path': '/api/download-sample',
                'method': 'GET',
                'description': 'Download generated sample data',
                'parameters': [
                    {'name': 'file', 'type': 'string', 'required': True, 'description': 'File identifier returned by sample-data endpoint'}
                ],
                'response': {'status': 'success', 'data': 'CSV content as string'}
            },
            {
                'path': '/api/categories',
                'method': 'GET',
                'description': 'Get list of allowed spending categories',
                'parameters': [],
                'response': {'status': 'success', 'categories': 'array of category strings'}
            }
        ],
        'data_formats': {
            'spending_csv': 'CSV with columns: date (YYYY-MM-DD), amount (numeric), category (string)',
            'income_csv': 'CSV with columns: date (YYYY-MM-DD), amount (numeric)',
            'spending_json': [{'date': 'YYYY-MM-DD', 'amount': 123.45, 'category': 'dining'}],
            'income_json': [{'date': 'YYYY-MM-DD', 'amount': 3000.00}]
        }
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    # Ensure the cache file exists
    if not os.path.exists(CACHE_FILE):
        save_cache({})
    
    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)