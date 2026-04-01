"""Backtest price prediction models"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.prediction.pipeline import PredictionPipeline
from src.data.database import Database
from src.config import Config
import pandas as pd

def backtest_all():
    """Backtest all models"""
    print("="*80)
    print("📊 BACKTESTING PRICE PREDICTION MODELS")
    print("="*80)
    
    db = Database(Config.DB_PATH)
    pipeline = PredictionPipeline(db)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
    
    results = []
    
    for symbol in symbols:
        print(f"\n🔍 Backtesting {symbol}...")
        
        result = pipeline.backtest_model(symbol, days_back=60, test_size=0.2)
        
        if 'error' not in result:
            results.append({
                'symbol': symbol,
                'accuracy': result.get('accuracy', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'test_samples': result.get('test_samples', 0)
            })
            
            print(f"  ✅ Accuracy: {result.get('accuracy', 0):.2%}")
            print(f"  ✅ Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"  ✅ Test Samples: {result.get('test_samples', 0)}")
            
            if result.get('precision'):
                print(f"  ✅ Precision: {result.get('precision', 0):.2%}")
                print(f"  ✅ Recall: {result.get('recall', 0):.2%}")
        else:
            print(f"  ❌ Error: {result.get('error')}")
    
    # Summary
    print("\n" + "="*80)
    print("📈 BACKTEST SUMMARY")
    print("="*80)
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        avg_accuracy = df['accuracy'].mean()
        print(f"\nAverage Accuracy: {avg_accuracy:.2%}")
        
        if avg_accuracy > 0.55:
            print("✅ Model meets >55% accuracy requirement!")
        else:
            print(f"⚠️ Model accuracy {avg_accuracy:.2%} is below 55% target")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    backtest_all()