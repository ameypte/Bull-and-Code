import os
import pandas as pd
import numpy as np

def calculate_rsi(data, periods=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    rsi = 100.0 - (100.0 / (1.0 + ma_up / ma_down))
    return rsi

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    macd = fast_ema - slow_ema
    signal_line = calculate_ema(macd, signal_period)
    histogram = macd - signal_line
    return macd

def process_stock_data(df):
    # Clean Volume column if it has commas
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
    
    # Calculate Technical Indicators
    df['EMA_9'] = calculate_ema(df['Close'], 9)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    
    df['SMA_9'] = calculate_sma(df['Close'], 9)
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    return df

def process_all_files(input_folder, output_folder):
    """
    Process all CSV files in a folder and save with technical indicators.
    
    Parameters:
    input_folder (str): Path to input folder containing CSV files.
    output_folder (str): Path to save processed CSV files.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_with_indicators.csv")
            
            # Process the data
            print(f"Processing {filename}...")
            df = pd.read_csv(input_path, parse_dates=['Date'])
            processed_df = process_stock_data(df)
            
            # Save processed data
            processed_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

# Example usage
if __name__ == "__main__":
    input_folder = "data"  # Folder containing input CSV files
    output_folder = "processed_data"  # Folder to save output files
    
    # Process all files
    process_all_files(input_folder, output_folder)
    print("All files have been processed and saved.")
