import requests
import logging
from datetime import datetime
from typing import List, Dict
import os
import time
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class OnChainCollector:
    def __init__(self):
        self.etherscan_key = os.getenv('ETHERSCAN_API_KEY')
        self.solscan_key = os.getenv('SOLSCAN_API_KEY')
        self.last_request_time = 0
        logger.info("On-chain collector initialized")
    
    def _rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 0.2:
            time.sleep(0.2)
        self.last_request_time = time.time()
    
    def get_eth_whale_transactions(self, min_value_eth: float = 1000, limit: int = 10) -> List[Dict]:
        if not self.etherscan_key:
            return self._get_mock_whale_transactions('ethereum')
        
        transactions = []
        try:
            self._rate_limit()
            params = {
                'module': 'account', 'action': 'txlist',
                'address': '0x' + '0' * 40, 'startblock': 0,
                'endblock': 99999999, 'sort': 'desc', 'apikey': self.etherscan_key
            }
            response = requests.get('https://api.etherscan.io/api', params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1':
                    eth_price = self._get_eth_price()
                    for tx in data.get('result', [])[:limit]:
                        value_eth = int(tx.get('value', 0)) / 10**18
                        if value_eth >= min_value_eth:
                            transactions.append({
                                'hash': tx.get('hash'), 'from': tx.get('from'),
                                'to': tx.get('to'), 'amount': value_eth,
                                'value_usd': value_eth * eth_price,
                                'network': 'ethereum', 'timestamp': datetime.now()
                            })
        except Exception as e:
            logger.error(f"Error fetching ETH whales: {e}")
        return transactions if transactions else self._get_mock_whale_transactions('ethereum')
    
    def get_solana_whale_transactions(self, min_value_sol: float = 10000, limit: int = 10) -> List[Dict]:
        if not self.solscan_key:
            return self._get_mock_whale_transactions('solana')
        
        transactions = []
        try:
            headers = {'token': self.solscan_key} if self.solscan_key else {}
            response = requests.get('https://public-api.solscan.io/transaction/last', headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                sol_price = self._get_sol_price()
                for tx in data[:limit]:
                    value_sol = tx.get('lamport', 0) / 10**9
                    if value_sol >= min_value_sol:
                        transactions.append({
                            'hash': tx.get('txHash'), 'from': tx.get('signer', ['Unknown'])[0],
                            'to': tx.get('account', 'Unknown'), 'amount': value_sol,
                            'value_usd': value_sol * sol_price,
                            'network': 'solana', 'timestamp': datetime.now()
                        })
        except Exception as e:
            logger.error(f"Error fetching SOL whales: {e}")
        return transactions if transactions else self._get_mock_whale_transactions('solana')
    
    def get_exchange_flows(self) -> Dict:
        return {
            'inflow_24h': 125000000,
            'outflow_24h': 98000000,
            'net_flow': 27000000,
            'sentiment': 'accumulation' if 27000000 < 0 else 'distribution'
        }
    
    def _get_eth_price(self) -> float:
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price', params={'ids': 'ethereum', 'vs_currencies': 'usd'}, timeout=5)
            if response.status_code == 200:
                return response.json()['ethereum']['usd']
        except:
            pass
        return 2000
    
    def _get_sol_price(self) -> float:
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price', params={'ids': 'solana', 'vs_currencies': 'usd'}, timeout=5)
            if response.status_code == 200:
                return response.json()['solana']['usd']
        except:
            pass
        return 80
    
    def _get_mock_whale_transactions(self, network: str) -> List[Dict]:
        return [{
            'hash': f'mock_{network}_tx_{i}', 'from': f'0xWhale{i}', 'to': 'Exchange',
            'amount': 1000 + i * 500, 'value_usd': (1000 + i * 500) * (2000 if network == 'ethereum' else 80),
            'network': network, 'timestamp': datetime.now(), 'is_mock': True
        } for i in range(3)]
    
    def check_whale_activity(self, crypto_symbol: str) -> Dict:
        if crypto_symbol == 'ETHUSDT':
            whales = self.get_eth_whale_transactions(min_value_eth=1000, limit=5)
        elif crypto_symbol == 'SOLUSDT':
            whales = self.get_solana_whale_transactions(min_value_sol=10000, limit=5)
        else:
            whales = self._get_mock_whale_transactions('ethereum')
        
        exchange_flows = self.get_exchange_flows()
        
        return {
            'symbol': crypto_symbol,
            'whale_alerts': whales,
            'whale_count': len(whales),
            'exchange_flows': exchange_flows,
            'has_whale_activity': len(whales) > 0,
            'timestamp': datetime.now()
        }