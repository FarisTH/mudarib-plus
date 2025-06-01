"""
Binance API - مجاني بالكامل للبيانات العامة
لا يحتاج تسجيل أو API key
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class BinanceAPI:
    """
    واجهة برمجة تطبيقات Binance للعملات الرقمية
    مجانية بالكامل - لا تحتاج API key
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.last_request_time = 0
        self.min_interval = 0.1  # 100ms بين الطلبات (آمن)
        
        # خريطة الرموز الشائعة
        self.symbol_map = {
            'BTC': 'BTCUSDT',
            'BITCOIN': 'BTCUSDT',
            'ETH': 'ETHUSDT', 
            'ETHEREUM': 'ETHUSDT',
            'BNB': 'BNBUSDT',
            'ADA': 'ADAUSDT',
            'CARDANO': 'ADAUSDT',
            'XRP': 'XRPUSDT',
            'RIPPLE': 'XRPUSDT',
            'DOT': 'DOTUSDT',
            'POLKADOT': 'DOTUSDT',
            'UNI': 'UNIUSDT',
            'UNISWAP': 'UNIUSDT',
            'LTC': 'LTCUSDT',
            'LITECOIN': 'LTCUSDT',
            'LINK': 'LINKUSDT',
            'CHAINLINK': 'LINKUSDT',
            'BCH': 'BCHUSDT',
            'DOGE': 'DOGEUSDT',
            'DOGECOIN': 'DOGEUSDT',
            'MATIC': 'MATICUSDT',
            'POLYGON': 'MATICUSDT',
            'SOL': 'SOLUSDT',
            'SOLANA': 'SOLUSDT',
            'AVAX': 'AVAXUSDT',
            'AVALANCHE': 'AVAXUSDT'
        }
    
    def _normalize_symbol(self, symbol: str) -> str:
        """تحويل رمز العملة إلى صيغة Binance"""
        symbol = symbol.upper().strip()
        
        # إذا كان الرمز في الخريطة
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        
        # إذا كان ينتهي بـ USDT بالفعل
        if symbol.endswith('USDT'):
            return symbol
            
        # إضافة USDT افتراضياً
        return f"{symbol}USDT"
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """إرسال طلب إلى Binance مع rate limiting"""
        try:
            # تطبيق rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            
            # إرسال الطلب
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params or {}, timeout=10)
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"خطأ Binance {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"خطأ في طلب Binance: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """الحصول على السعر الحالي لعملة رقمية"""
        try:
            binance_symbol = self._normalize_symbol(symbol)
            
            data = self._make_request("/api/v3/ticker/price", {
                'symbol': binance_symbol
            })
            
            if data and 'price' in data:
                price = float(data['price'])
                logger.info(f"السعر الحالي لـ {symbol}: ${price:,.2f}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"خطأ في جلب سعر {symbol}: {str(e)}")
            return None
    
    def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """الحصول على إحصائيات 24 ساعة"""
        try:
            binance_symbol = self._normalize_symbol(symbol)
            
            data = self._make_request("/api/v3/ticker/24hr", {
                'symbol': binance_symbol
            })
            
            if not data:
                return None
            
            stats = {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChange']),
                'change_percent_24h': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'volume_usd_24h': float(data['quoteVolume']),
                'open_price': float(data['openPrice']),
                'trades_count': int(data['count'])
            }
            
            logger.info(f"إحصائيات {symbol}: تغيير 24س {stats['change_percent_24h']:.2f}%")
            return stats
            
        except Exception as e:
            logger.error(f"خطأ في جلب إحصائيات {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, interval: str = "1d", 
                          limit: int = 100) -> Optional[pd.DataFrame]:
        """
        جلب البيانات التاريخية (الشموع اليابانية)
        
        Args:
            symbol: رمز العملة
            interval: الفترة الزمنية (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
            limit: عدد الشموع (حد أقصى 1000)
        """
        try:
            binance_symbol = self._normalize_symbol(symbol)
            
            # التحقق من صحة الفترة الزمنية
            valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', 
                             '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            
            if interval not in valid_intervals:
                logger.error(f"فترة زمنية غير صحيحة: {interval}")
                return None
            
            # تحديد العدد (حد أقصى 1000)
            limit = min(max(1, limit), 1000)
            
            data = self._make_request("/api/v3/klines", {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            })
            
            if not data:
                return None
            
            # تحويل إلى DataFrame
            columns = [
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # تحويل الوقت
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # تحويل أعمدة الأسعار إلى float
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # اختيار الأعمدة المطلوبة فقط
            df = df[price_columns].copy()
            
            # تنظيف البيانات
            df = df.dropna()
            
            logger.info(f"تم جلب {len(df)} شمعة من {symbol} بفترة {interval}")
            return df
            
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات التاريخية لـ {symbol}: {str(e)}")
            return None
    
    def get_market_data_by_period(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        جلب البيانات بناءً على فترة زمنية مفهومة
        
        Args:
            symbol: رمز العملة
            period: الفترة (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)
        """
        try:
            # تحويل period إلى interval و limit
            period_config = {
                '1d': {'interval': '1h', 'limit': 24},
                '5d': {'interval': '4h', 'limit': 30},
                '1mo': {'interval': '1d', 'limit': 30},
                '3mo': {'interval': '1d', 'limit': 90},
                '6mo': {'interval': '1d', 'limit': 180},
                '1y': {'interval': '1d', 'limit': 365},
                '2y': {'interval': '1d', 'limit': 730},
                '5y': {'interval': '1w', 'limit': 260}  # 5 سنوات بالأسابيع
            }
            
            config = period_config.get(period)
            if not config:
                logger.error(f"فترة غير مدعومة: {period}")
                return None
            
            return self.get_historical_data(
                symbol=symbol,
                interval=config['interval'], 
                limit=config['limit']
            )
            
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات {symbol} للفترة {period}: {str(e)}")
            return None
    
    def get_top_cryptocurrencies(self, limit: int = 50) -> List[Dict]:
        """الحصول على أفضل العملات الرقمية حسب الحجم"""
        try:
            data = self._make_request("/api/v3/ticker/24hr")
            
            if not data:
                return []
            
            # تصفية العملات مقابل USDT فقط
            usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
            
            # ترتيب حسب حجم التداول
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            
            # أخذ أفضل العملات
            top_cryptos = []
            for i, item in enumerate(usdt_pairs[:limit]):
                crypto = {
                    'rank': i + 1,
                    'symbol': item['symbol'].replace('USDT', ''),
                    'full_symbol': item['symbol'],
                    'price': float(item['lastPrice']),
                    'change_24h': float(item['priceChangePercent']),
                    'volume_usd': float(item['quoteVolume']),
                    'trades_count': int(item['count'])
                }
                top_cryptos.append(crypto)
            
            logger.info(f"تم جلب أفضل {len(top_cryptos)} عملة رقمية")
            return top_cryptos
            
        except Exception as e:
            logger.error(f"خطأ في جلب أفضل العملات: {str(e)}")
            return []
    
    def search_cryptocurrency(self, query: str) -> List[Dict]:
        """البحث عن عملة رقمية"""
        try:
            # جلب جميع الرموز
            data = self._make_request("/api/v3/ticker/24hr")
            
            if not data:
                return []
            
            query = query.upper().strip()
            results = []
            
            for item in data:
                symbol = item['symbol']
                
                # البحث في الرموز التي تحتوي على الاستعلام
                if (query in symbol and symbol.endswith('USDT') and 
                    len(symbol.replace('USDT', '')) >= 2):
                    
                    crypto_symbol = symbol.replace('USDT', '')
                    result = {
                        'symbol': crypto_symbol,
                        'full_symbol': symbol,
                        'price': float(item['lastPrice']),
                        'change_24h': float(item['priceChangePercent']),
                        'volume_usd': float(item['quoteVolume'])
                    }
                    results.append(result)
            
            # ترتيب النتائج حسب الحجم
            results.sort(key=lambda x: x['volume_usd'], reverse=True)
            
            logger.info(f"تم العثور على {len(results)} نتيجة للبحث عن '{query}'")
            return results[:10]  # أفضل 10 نتائج
            
        except Exception as e:
            logger.error(f"خطأ في البحث عن {query}: {str(e)}")
            return []
    
    def get_exchange_info(self) -> Optional[Dict]:
        """الحصول على معلومات البورصة والرموز المدعومة"""
        try:
            data = self._make_request("/api/v3/exchangeInfo")
            
            if not data:
                return None
            
            # استخراج معلومات مفيدة
            info = {
                'timezone': data.get('timezone'),
                'server_time': data.get('serverTime'),
                'symbols_count': len(data.get('symbols', [])),
                'usdt_pairs': []
            }
            
            # جمع الأزواج مع USDT
            for symbol_info in data.get('symbols', []):
                if (symbol_info['symbol'].endswith('USDT') and 
                    symbol_info['status'] == 'TRADING'):
                    
                    info['usdt_pairs'].append({
                        'symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status']
                    })
            
            logger.info(f"معلومات البورصة: {info['symbols_count']} رمز، {len(info['usdt_pairs'])} زوج USDT")
            return info
            
        except Exception as e:
            logger.error(f"خطأ في جلب معلومات البورصة: {str(e)}")
            return None
    
    def is_symbol_valid(self, symbol: str) -> bool:
        """التحقق من صحة رمز العملة"""
        try:
            binance_symbol = self._normalize_symbol(symbol)
            
            # محاولة جلب السعر
            data = self._make_request("/api/v3/ticker/price", {
                'symbol': binance_symbol
            })
            
            return data is not None and 'price' in data
            
        except Exception as e:
            logger.error(f"خطأ في التحقق من {symbol}: {str(e)}")
            return False
    
    def get_supported_symbols(self) -> List[str]:
        """الحصول على قائمة الرموز المدعومة"""
        try:
            exchange_info = self.get_exchange_info()
            
            if not exchange_info:
                return []
            
            symbols = []
            for pair in exchange_info['usdt_pairs']:
                symbols.append(pair['base_asset'])
            
            return sorted(list(set(symbols)))
            
        except Exception as e:
            logger.error(f"خطأ في جلب الرموز المدعومة: {str(e)}")
            return []

# دالة مساعدة للاستخدام السريع
def get_crypto_data(symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
    """
    دالة سريعة لجلب بيانات العملة الرقمية
    
    الاستخدام:
        data = get_crypto_data("BTC", "1mo")
    """
    api = BinanceAPI()
    return api.get_market_data_by_period(symbol, period)

def get_crypto_price(symbol: str) -> Optional[float]:
    """
    دالة سريعة لجلب السعر الحالي
    
    الاستخدام:
        price = get_crypto_price("BTC")
    """
    api = BinanceAPI()
    return api.get_current_price(symbol)