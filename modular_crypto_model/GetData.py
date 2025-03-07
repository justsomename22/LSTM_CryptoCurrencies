import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from data_processing import add_technical_indicators

base_url = "https://api.coingecko.com/api/v3"
crypto_tickers = ["bitcoin", "ethereum", "monero"]

def fetch_crypto_historical_data(crypto_id, days=365):
    """
    Fetch historical market data for a specific cryptocurrency.

    Parameters:
        crypto_id (str): The ID of the cryptocurrency (e.g., 'bitcoin').
        days (int): Number of days of historical data to fetch.

    Returns:
        dict: JSON response containing historical market data.
    """
    url = f"{base_url}/coins/{crypto_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def process_crypto_data(crypto_id, historical_data):
    """
    Process the historical data for a cryptocurrency into a structured format.

    Parameters:
        crypto_id (str): The ID of the cryptocurrency.
        historical_data (dict): Raw historical data from the API.

    Returns:
        list: List of dictionaries containing processed data.
    """
    processed_data = []
    if 'prices' in historical_data:
        for price in historical_data['prices']:
            date = datetime.fromtimestamp(price[0] / 1000).strftime('%Y-%m-%d')
            processed_data.append({"crypto_id": crypto_id, "date": date, "price": price[1]})
    return processed_data

if __name__ == "__main__":
    csv_file = "cryptocurrency_data.csv"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        # Initial fetch: get 365 days of data
        all_crypto_data = []
        for crypto_id in crypto_tickers:
            data = fetch_crypto_historical_data(crypto_id, days=365)
            processed = process_crypto_data(crypto_id, data)
            all_crypto_data.extend(processed)
            time.sleep(2)  # Rate limit for API
        df = pd.DataFrame(all_crypto_data)
        df = add_technical_indicators(df)
        df.to_csv(csv_file, index=False)
        print("Initial data saved to", csv_file)
    else:
        # Load existing data
        existing_df = pd.read_csv(csv_file)
        
        # Update with new data
        for crypto_id in crypto_tickers:
            if crypto_id in existing_df['crypto_id'].unique():
                # Find the latest date for this crypto
                latest_date_str = existing_df[existing_df['crypto_id'] == crypto_id]['date'].max()
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                today = datetime.now().date()
                
                # Check if new data is needed
                if latest_date.date() < today:
                    days_to_fetch = (today - latest_date.date()).days
                    data = fetch_crypto_historical_data(crypto_id, days=days_to_fetch)
                    processed = process_crypto_data(crypto_id, data)
                    new_df = pd.DataFrame(processed)
                    
                    # Filter out dates already in CSV
                    new_df = new_df[new_df['date'] > latest_date_str]
                    if not new_df.empty:
                        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Crypto not in CSV, fetch full history
                data = fetch_crypto_historical_data(crypto_id, days=365)
                processed = process_crypto_data(crypto_id, data)
                new_df = pd.DataFrame(processed)
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            time.sleep(2)  # Rate limit for API
        
        # Sort the DataFrame for consistency
        existing_df = existing_df.sort_values(by=['crypto_id', 'date']).reset_index(drop=True)
        # Recalculate technical indicators
        existing_df = add_technical_indicators(existing_df)
        # Save updated data
        existing_df.to_csv(csv_file, index=False)
        print("Updated data saved to", csv_file)

# Pseudo-code for scheduling (use cron or `schedule` library):
# import schedule
# schedule.every().day.at("00:00").do(lambda: os.system("python GetData.py"))