import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# --- Configuration ---
# Base URL for CoinGecko API
base_url = "https://api.coingecko.com/api/v3"

# List of cryptocurrency symbols (using CoinGecko IDs)
crypto_tickers = [
    "bitcoin",
    "ethereum",
    "litecoin",
    "ripple",
    "cardano"
]

# Set start date to 365 days ago to comply with CoinGecko's free tier limitations
end_date = datetime.now()  # Current date
start_date = end_date - timedelta(days=364)  # Use 364 to be safe

# Format dates as strings
start_date_str = start_date.strftime("%Y-%m-%d")  # Start date in YYYY-MM-DD format
end_date_str = end_date.strftime("%Y-%m-%d")  # End date in YYYY-MM-DD format

# File to save the cryptocurrency data
csv_filename = "C:/Users/ismas/Python Machine Learning Crypto/cryptocurrency_data.csv"

# Optional: Add API key if you have one (for higher rate limits)
api_key = None  # Replace with your API key if you have one

# --- Functions ---

def fetch_crypto_historical_data(crypto_id, start_date, end_date, max_retries=3):
    """
    Fetches historical price and volume data for a cryptocurrency from CoinGecko API.
    Includes retry logic with exponential backoff.
    
    Args:
        crypto_id (str): CoinGecko ID of the cryptocurrency (e.g., 'bitcoin').
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        max_retries (int): Maximum number of retry attempts.
        
    Returns:
        dict or None: Dictionary containing historical data, or None if error.
    """
    # Convert string dates to datetime objects first
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # Convert start date to datetime
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")  # Convert end date to datetime
    
    # Convert to Unix timestamps
    start_timestamp = int(start_dt.timestamp())  # Convert start date to Unix timestamp
    end_timestamp = int(end_dt.timestamp())  # Convert end date to Unix timestamp
    
    url = f"{base_url}/coins/{crypto_id}/market_chart/range"  # Construct API URL
    params = {
        "vs_currency": "usd",  # Currency to compare against
        "from": start_timestamp,  # Start timestamp
        "to": end_timestamp  # End timestamp
    }
    
    # Add API key to headers if available
    headers = {}
    if api_key:
        headers["x-cg-api-key"] = api_key  # Include API key in headers
    
    print(f"Requesting data from {start_date} to {end_date} for {crypto_id}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)  # Make API request
            status_code = response.status_code  # Get response status code
            
            print(f"Attempt {attempt+1}/{max_retries} - Status code: {status_code}")
            
            if status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', 2 ** attempt))  # Get retry time
                print(f"Rate limit hit. Waiting {retry_after} seconds before retry.")
                time.sleep(retry_after)  # Wait before retrying
                continue
                
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()  # Parse JSON response
            print(f"Successfully retrieved data for {crypto_id}")
            return data  # Return the retrieved data
            
        except requests.exceptions.Timeout:
            wait_time = 2 ** attempt  # Exponential backoff for timeouts
            print(f"Timeout error. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait before retrying
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                print(f"Bad request for {crypto_id}: {response.text}")
                return None  # Don't retry for bad requests
            
            wait_time = 2 ** attempt  # Exponential backoff for HTTP errors
            print(f"HTTP error {response.status_code}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait before retrying
            
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt  # Exponential backoff for other request errors
            print(f"Request error: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait before retrying
    
    # If we've exhausted all retries
    print(f"Failed to fetch data for {crypto_id} after {max_retries} attempts.")
    return None  # Return None if all attempts fail

def process_crypto_data(crypto_id, historical_data):
    """
    Processes the historical data from CoinGecko API and extracts relevant information.
    
    Args:
        crypto_id (str): CoinGecko ID of the cryptocurrency.
        historical_data (dict): Data from CoinGecko API.
        
    Returns:
        list: List of dictionaries, each containing processed data for a day.
    """
    processed_data = []  # Initialize list to hold processed data
    if historical_data and 'prices' in historical_data and 'total_volumes' in historical_data:
        prices = historical_data['prices']  # Extract prices
        volumes = historical_data['total_volumes']  # Extract volumes
        
        # Ensure prices and volumes lists are of the same length
        data_points = min(len(prices), len(volumes))  # Get the minimum length
        
        for i in range(data_points):
            timestamp = prices[i][0]  # Timestamp is in milliseconds
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')  # Convert to date string
            price = prices[i][1]  # Get price
            volume = volumes[i][1]  # Get volume
            
            processed_data.append({
                "crypto_id": crypto_id,  # Add cryptocurrency ID
                "date": date,  # Add date
                "price": price,  # Add price
                "volume": volume  # Add volume
            })
    else:
        print(f"No valid historical data found for {crypto_id}")  # Handle case with no data
    return processed_data  # Return processed data

# --- Main Script ---

all_crypto_data = []  # List to hold all cryptocurrency data
print(f"Fetching cryptocurrency data for {len(crypto_tickers)} cryptos...")
print(f"Time range: {start_date_str} to {end_date_str} (within 365-day limit)")

for crypto_id in crypto_tickers:
    print(f"Fetching data for {crypto_id}...")
    historical_data = fetch_crypto_historical_data(crypto_id, start_date_str, end_date_str)  # Fetch historical data
    
    if historical_data:
        processed_data = process_crypto_data(crypto_id, historical_data)  # Process the fetched data
        all_crypto_data.extend(processed_data)  # Add processed data to the main list
        print(f"Successfully processed data for {crypto_id} - {len(processed_data)} data points")
    
    # Be respectful to the API, especially for free tier
    wait_time = 6 if not api_key else 2  # Longer wait time for free tier
    print(f"Waiting {wait_time} seconds before next request...")
    time.sleep(wait_time)  # Wait before the next request

# Create DataFrame and save to CSV
df = pd.DataFrame(all_crypto_data)  # Create DataFrame from the collected data
if not df.empty:
    try:
        df.to_csv(csv_filename, index=False)  # Save DataFrame to CSV
        print(f"Cryptocurrency data saved to '{csv_filename}'. Rows: {len(df)}")
        
        # Print a sample of the data
        print("\nData sample:")
        print(df.head())  # Display the first few rows of the DataFrame
    except PermissionError:
        print(f"Error: Permission denied when saving to {csv_filename}")  # Handle permission errors
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")  # Handle other exceptions
else:
    print("No cryptocurrency data fetched and processed.")  # Handle case with no data

    