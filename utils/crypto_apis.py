"""
مجموعة APIs مجانية للعملات الرقمية
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import yfinance as yf

logger = logging.getLogger(__name__)

class CryptoAPIManager:
    """مدير APIs العملات الرقمية مع تركيز على الخيارات المجانية"""
    
    def __init__(self):
        self.last_call_times = {}
        self.rate_limits = {
            'coincap': 1.0,      # ثانية واحدة بين الطلبات
            'binance': 0.1,      # عُشر ثانية بين الطلبات
            'cryptocompare': 1.0, # ثانية واحدة بين الطلبات
            'yfinance': 1.0      # ثانية واحدة بين الطلبات
        }
    
    def get_crypto_data(self, symbol: str, period: str = "1y", 
                       source: str = "auto") -> Optional[pd.DataFrame]:
        """
        جلب بيانات العملات الرقمية من أفضل مصدر مجاني
        
        Args:
            symbol: رمز العملة (BTC, ETH, etc.)
            period: الفترة الزمنية
            source: المصدر ("auto", "yfinance", "binance", "coincap")
        """
        try:
            if source == "auto":
                # جرب المصادر بالترتيب حسب الجودة
                for src in ["binance", "yfinance", "coincap"]:
                    data = self._get_data_from_source(symbol, period, src)
                    if data is not None and not data.empty:
                        return data
                return None
            else:
                return self._get_data_from_source(symbol, period, source)
                
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات {symbol}: {str(e)}")
            return None
    
    def _get_data_from_source(self, symbol: str, period: str, source: str) -> Optional[pd.DataFrame]:
        """جلب البيانات من مصدر محدد"""
        try:
            if source == "yfinance":
                return self._get_yfinance_crypto(symbol, period)
            elif source == "binance":
                return self._get_binance_data(symbol, period)
            elif source == "coincap":
                return self._get_coincap_data(symbol, period)
            elif source == "cryptocompare":
                return self._get_cryptocompare_data(symbol, period)
            else:
                logger.warning(f"مصدر غير معروف: {source}")
                return None
                
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات من {source}: {str(e)}")
            return None
    
    def _get_yfinance_crypto(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """جلب بيانات العملات الرقمية من Yahoo Finance"""
        try:
            self._apply_rate_limit('yfinance')
            
            # تحويل الرمز إلى صيغة Yahoo Finance
            if not symbol.endswith('-USD'):
                symbol = f"{symbol}-USD"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            # تنظيف البيانات
            data = data.dropna()
            
            logger.info(f"تم جلب بيانات {symbol} من Yahoo Finance")
            return data
            
        except Exception as e:
            logger.error(f"خطأ في جلب {symbol} من Yahoo Finance: {str(e)}")
            return None
    
    def _get_binance_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """جلب البيانات من Binance API (مجاني)"""
        try:
            self._apply_rate_limit('binance')
            
            # تحويل الرمز إلى صيغة Binance
            symbol_binance = f"{symbol}USDT"
            
            # تحويل period إلى صيغة Binance
            interval_map = {
                '1d': '1d',
                '5d': '1d', 
                '1mo': '1d',
                '3mo': '1d',
                '6mo': '1d',
                '1y': '1d',
                '2y': '1d',
                '5y': '1d'
            }
            
            interval = interval_map.get(period, '1d')
            
            # حساب عدد الشموع المطلوبة
            limit_map = {
                '1d': 1,
                '5d': 5,
                '1mo': 30,
                '3mo': 90,
                '6mo': 180,
                '1y': 365,
                '2y': 730,
                '5y': 1000  # الحد الأقصى لـ Binance
            }
            
            limit = limit_map.get(period, 365)
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol_binance,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance يحدد بـ 1000
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"خطأ Binance: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data:
                return None
            
            # تحويل إلى DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # تحويل الأنواع
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # اختيار الأعمدة المطلوبة وتحويلها لـ float
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[price_columns].astype(float)
            
            logger.info(f"تم جلب بيانات {symbol} من Binance")
            return df
            
        except Exception as e:
            logger.error(f"خطأ في جلب {symbol} من Binance: {str(e)}")
            return None
    
    def _get_coincap_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """جلب البيانات من CoinCap API (مجاني)"""
        try:
            self._apply_rate_limit('coincap')
            
            # البحث عن ID العملة
            search_url = "https://api.coincap.io/v2/assets"
            search_params = {'search': symbol}
            
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            
            if not search_data.get('data'):
                logger.error(f"لم يتم العثور على {symbol} في CoinCap")
                return None
            
            # أخذ أول نتيجة
            asset_id = search_data['data'][0]['id']
            
            # جلب البيانات التاريخية
            # CoinCap يوفر بيانات محدودة فقط
            url = f"https://api.coincap.io/v2/assets/{asset_id}/history"
            
            # تحويل period إلى ميللي ثانية
            period_map = {
                '1d': 'd1',
                '5d': 'd1', 
                '1mo': 'd1',
                '3mo': 'd1',
                '6mo': 'd1',
                '1y': 'd1'
            }
            
            interval = period_map.get(period, 'd1')
            
            params = {
                'interval': interval
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"خطأ CoinCap: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('data'):
                return None
            
            # تحويل إلى DataFrame
            df = pd.DataFrame(data['data'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.set_index('time')
            
            # CoinCap يوفر فقط سعر الإغلاق
            df['priceUsd'] = df['priceUsd'].astype(float)
            
            # تقدير OHLC من سعر الإغلاق
            df['Close'] = df['priceUsd']
            df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            df['High'] = df[['Open', 'Close']].max(axis=1) * 1.01
            df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.99
            df['Volume'] = 0  # CoinCap لا يوفر volume في التاريخية
            
            # اختيار الأعمدة المطلوبة
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # تصفية حسب الفترة
            if period != 'max':
                days = self._period_to_days(period)
                start_date = datetime.now() - timedelta(days=days)
                df = df[df.index >= start_date]
            
            logger.info(f"تم جلب بيانات {symbol} من CoinCap")
            return df
            
        except Exception as e:
            logger.error(f"خطأ في جلب {symbol} من CoinCap: {str(e)}")
            return None
    
    def _get_cryptocompare_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """جلب البيانات من CryptoCompare API (محدود مجاني)"""
        try:
            self._apply_rate_limit('cryptocompare')
            
            # تحديد نوع البيانات حسب الفترة
            if period in ['1d', '5d']:
                endpoint = 'histohour'
                limit = 24 if period == '1d' else 120
            else:
                endpoint = 'histoday'
                limit = min(self._period_to_days(period), 2000)
            
            url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
            params = {
                'fsym': symbol.upper(),
                'tsym': 'USD',
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"خطأ CryptoCompare: {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get('Response') == 'Error':
                logger.error(f"خطأ CryptoCompare: {data.get('Message')}")
                return None
            
            if not data.get('Data', {}).get('Data'):
                return None
            
            # تحويل إلى DataFrame
            df = pd.DataFrame(data['Data']['Data'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            
            # إعادة تسمية الأعمدة
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volumefrom': 'Volume'
            })
            
            # اختيار الأعمدة المطلوبة
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
            logger.info(f"تم جلب بيانات {symbol} من CryptoCompare")
            return df
            
        except Exception as e:
            logger.error(f"خطأ في جلب {symbol} من CryptoCompare: {str(e)}")
            return None
    
    def get_current_prices(self) -> Dict[str, float]:
        """جلب الأسعار الحالية للعملات الرقمية الرئيسية"""
        try:
            self._apply_rate_limit('binance')
            
            url = "https://api.binance.com/api/v3/ticker/price"
            response = requests.get(url)
            
            if response.status_code != 200:
                return {}
            
            data = response.json()
            
            # تصفية العملات الرئيسية مقابل USDT
            major_cryptos = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 
                           'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT']
            
            prices = {}
            for item in data:
                symbol = item['symbol']
                if symbol in major_cryptos:
                    crypto_name = symbol.replace('USDT', '')
                    prices[crypto_name] = float(item['price'])
            
            return prices
            
        except Exception as e:
            logger.error(f"خطأ في جلب الأسعار الحالية: {str(e)}")
            return {}
    
    def get_crypto_info(self, symbol: str) -> Optional[Dict]:
        """جلب معلومات العملة الرقمية"""
        try:
            # محاولة جلب من multiple sources
            info = {}
            
            # معلومات أساسية من Binance
            binance_info = self._get_binance_symbol_info(symbol)
            if binance_info:
                info.update(binance_info)
            
            # معلومات إضافية من CoinCap
            coincap_info = self._get_coincap_info(symbol)
            if coincap_info:
                info.update(coincap_info)
            
            return info if info else None
            
        except Exception as e:
            logger.error(f"خطأ في جلب معلومات {symbol}: {str(e)}")
            return None
    
    def _get_binance_symbol_info(self, symbol: str) -> Optional[Dict]:
        """جلب معلومات الرمز من Binance"""
        try:
            self._apply_rate_limit('binance')
            
            symbol_binance = f"{symbol}USDT"
            
            # معلومات السعر
            price_url = "https://api.binance.com/api/v3/ticker/24hr"
            price_params = {'symbol': symbol_binance}
            
            price_response = requests.get(price_url, params=price_params)
            
            if price_response.status_code != 200:
                return None
            
            price_data = price_response.json()
            
            return {
                'symbol': symbol,
                'price': float(price_data['lastPrice']),
                'change_24h': float(price_data['priceChangePercent']),
                'volume_24h': float(price_data['volume']),
                'high_24h': float(price_data['highPrice']),
                'low_24h': float(price_data['lowPrice'])
            }
            
        except Exception as e:
            logger.error(f"خطأ في جلب معلومات Binance: {str(e)}")
            return None
    
    def _get_coincap_info(self, symbol: str) -> Optional[Dict]:
        """جلب معلومات إضافية من CoinCap"""
        try:
            self._apply_rate_limit('coincap')
            
            # البحث عن العملة
            search_url = "https://api.coincap.io/v2/assets"
            search_params = {'search': symbol, 'limit': 1}
            
            response = requests.get(search_url, params=search_params)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data.get('data'):
                return None
            
            asset = data['data'][0]
            
            return {
                'name': asset.get('name'),
                'market_cap': float(asset.get('marketCapUsd', 0)),
                'supply': float(asset.get('supply', 0)),
                'max_supply': float(asset.get('maxSupply', 0)) if asset.get('maxSupply') else None,
                'rank': int(asset.get('rank', 0))
            }
            
        except Exception as e:
            logger.error(f"خطأ في جلب معلومات CoinCap: {str(e)}")
            return None
    
    def _apply_rate_limit(self, source: str) -> None:
        """تطبيق حدود الاستخدام"""
        if source not in self.rate_limits:
            return
            
        current_time = time.time()
        last_call = self.last_call_times.get(source, 0)
        time_diff = current_time - last_call
        min_interval = self.rate_limits[source]
        
        if time_diff < min_interval:
            sleep_time = min_interval - time_diff
            time.sleep(sleep_time)
        
        self.last_call_times[source] = time.time()
    
    def _period_to_days(self, period: str) -> int:
        """تحويل فترة نصية إلى أيام"""
        period_map = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        return period_map.get(period, 365)
    
    def get_supported_cryptos(self) -> List[str]:
        """الحصول على قائمة العملات الرقمية المدعومة"""
        return [
            'BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOT', 'UNI', 'LTC',
            'LINK', 'BCH', 'XLM', 'USDC', 'DOGE', 'USDT', 'WBTC',
            'AAVE', 'EOS', 'XMR', 'TRX', 'XTZ', 'ATOM', 'NEO',
            'VET', 'IOTA', 'DASH', 'ZEC', 'QTUM', 'OMG', 'BAT'
        ]