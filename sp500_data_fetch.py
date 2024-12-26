import yfinance as yf
from datetime import datetime

# Define start date
start_date = '1980-01-01'

# Get the current date dynamically
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch historical data for S&P 500 (^GSPC)
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)

# Save to CSV
csv_filename = f"sp500_data_{start_date}_to_{end_date}.csv"
sp500_data.to_csv(csv_filename)

print(f"S&P 500 data saved as {csv_filename}")