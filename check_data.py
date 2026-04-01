# Create check_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.database import Database
from src.config import Config
from datetime import datetime, timedelta

db = Database(Config.DB_PATH)

print("Checking database for historical data...")
print("="*60)

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']

for symbol in symbols:
    latest = db.get_latest_timestamp(symbol)
    if latest:
        print(f"\n{symbol}:")
        print(f"  Latest data: {latest}")
        
        # Check how many days of data you have
        start_date = latest - timedelta(days=90)
        df = db.get_historical_data(symbol, start_date, latest)
        print(f"  Records in last 90 days: {len(df)}")
        
        if len(df) > 0:
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print(f"\n{symbol}: No data found!")