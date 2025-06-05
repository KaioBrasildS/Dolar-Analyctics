import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_usd_brl_historical_series(start_date, end_date):
    """
    Fetches the historical exchange rate series from USD to BRL using the AwesomeAPI.

    Parameters:
    - start_date: Start date in the format 'YYYY-MM-DD'.
    - end_date: End date in the format 'YYYY-MM-DD'.

    Returns:
    - DataFrame containing the exchange rates and dates.
    """
    base_url = "https://economia.awesomeapi.com.br/json/daily/USD-BRL/365"
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    current_date = end_date
    complete_data = []

    while current_date >= start_date:
        # Adjust the interval to up to 365 days
        end_date_str = current_date.strftime("%Y%m%d")
        start_date_str = max(start_date, current_date - timedelta(days=364)).strftime("%Y%m%d")

        # Build URL with the date range
        url = f"{base_url}?start_date={start_date_str}&end_date={end_date_str}"

        # Make the request
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            complete_data.extend(data)
        else:
            print(f"Request error: {response.status_code}")

        # Update the current date
        current_date -= timedelta(days=365)

    # Organize the data into a DataFrame
    if complete_data:
        df = pd.DataFrame(complete_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # Convert timestamps
        df = df.sort_values("timestamp")  # Sort by date
        df = df[["timestamp", "bid", "ask"]]  # Select relevant columns
        df.columns = ["Date", "Bid (USD-BRL)", "Ask (USD-BRL)"]
        return df
    else:
        print("No data found.")
        return None
