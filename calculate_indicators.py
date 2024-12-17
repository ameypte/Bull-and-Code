import pandas as pd
import numpy as np
import talib

def calculate_technical_indicators(csv_path):
    """
    Calculate technical indicators for a stock price dataset.
    
    Parameters:
    csv_path (str): Path to the input CSV file
    
    Returns:
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # Clean Volume column (remove commas)
    df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
    
    # Calculate Exponential Moving Averages (EMA)
    df['EMA_9'] = talib.EMA(df['Close'], timeperiod=9)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    
    # Calculate Simple Moving Averages (SMA)
    df['SMA_9'] = talib.SMA(df['Close'], timeperiod=9)
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    # Calculate MACD
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Histogram'] = hist
    
    return df

def save_processed_data(df, output_path):
    """
    Save the processed DataFrame to a new CSV file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with technical indicators
    output_path (str): Path to save the output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_path = "data\Adani Enterprises.csv"  # Replace with your actual input file path
    output_path = "data\Adani Enterprises_with_indicators.csv"  # Replace with your desired output file path
    
    # Process the data
    processed_df = calculate_technical_indicators(input_path)
    
    # Save the processed data
    save_processed_data(processed_df, output_path)
    
    # Display the first few rows to verify
    print(processed_df.head())