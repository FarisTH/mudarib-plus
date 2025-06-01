"""
اختبار Binance API - لا يحتاج أي مفاتيح!
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.binance_api import BinanceAPI, get_crypto_data, get_crypto_price
import pandas as pd

def test_binance_api():
    """اختبار شامل لـ Binance API"""
    
    print("🚀 اختبار Binance API - مجاني بالكامل!")
    print("="*50)
    
    # إنشاء كائن API
    api = BinanceAPI()
    
    # 1. اختبار السعر الحالي
    print("\n📊 اختبار الأسعار الحالية:")
    test_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP']
    
    for symbol in test_symbols:
        price = api.get_current_price(symbol)
        if price:
            print(f"   {symbol}: ${price:,.2f}")
        else:
            print(f"   {symbol}: فشل في جلب السعر")
    
    # 2. اختبار إحصائيات 24 ساعة
    print("\n📈 اختبار إحصائيات 24 ساعة لـ Bitcoin:")
    btc_stats = api.get_24h_stats('BTC')
    if btc_stats:
        print(f"   السعر: ${btc_stats['price']:,.2f}")
        print(f"   التغيير 24س: {btc_stats['change_percent_24h']:+.2f}%")
        print(f"   الحد الأعلى: ${btc_stats['high_24h']:,.2f}")
        print(f"   الحد الأدنى: ${btc_stats['low_24h']:,.2f}")
        print(f"   الحجم: ${btc_stats['volume_usd_24h']:,.0f}")
    else:
        print("   فشل في جلب إحصائيات BTC")
    
    # 3. اختبار البيانات التاريخية
    print("\n📜 اختبار البيانات التاريخية لـ Ethereum:")
    eth_data = api.get_historical_data('ETH', '1d', 30)  # 30 يوم
    if eth_data is not None:
        print(f"   تم جلب {len(eth_data)} يوم من البيانات")
        print(f"   أحدث سعر إغلاق: ${eth_data['Close'].iloc[-1]:,.2f}")
        print(f"   أعلى سعر: ${eth_data['High'].max():,.2f}")
        print(f"   أقل سعر: ${eth_data['Low'].min():,.2f}")
        print(f"   متوسط الحجم: {eth_data['Volume'].mean():,.0f}")
    else:
        print("   فشل في جلب البيانات التاريخية")
    
    # 4. اختبار البيانات بالفترات
    print("\n⏰ اختبار البيانات بفترات مختلفة:")
    periods = ['1d', '1mo', '3mo', '1y']
    
    for period in periods:
        data = api.get_market_data_by_period('BTC', period)
        if data is not None:
            days = len(data)
            latest_price = data['Close'].iloc[-1]
            print(f"   {period}: {days} نقطة بيانات، آخر سعر ${latest_price:,.2f}")
        else:
            print(f"   {period}: فشل في جلب البيانات")
    
    # 5. اختبار أفضل العملات
    print("\n🏆 اختبار أفضل 10 عملات رقمية:")
    top_cryptos = api.get_top_cryptocurrencies(10)
    if top_cryptos:
        for crypto in top_cryptos[:5]:  # أفضل 5 فقط للعرض
            print(f"   #{crypto['rank']} {crypto['symbol']}: "
                  f"${crypto['price']:,.2f} "
                  f"({crypto['change_24h']:+.1f}%)")
    else:
        print("   فشل في جلب أفضل العملات")
    
    # 6. اختبار البحث
    print("\n🔍 اختبار البحث عن عملات:")
    search_results = api.search_cryptocurrency('DOG')
    if search_results:
        print(f"   تم العثور على {len(search_results)} نتيجة للبحث عن 'DOG':")
        for result in search_results[:3]:  # أفضل 3 نتائج
            print(f"     {result['symbol']}: ${result['price']:,.4f} "
                  f"({result['change_24h']:+.1f}%)")
    else:
        print("   لم يتم العثور على نتائج")
    
    # 7. اختبار صحة الرموز
    print("\n✅ اختبار صحة الرموز:")
    test_symbols = ['BTC', 'ETH', 'INVALIDCOIN', 'DOGE']
    for symbol in test_symbols:
        is_valid = api.is_symbol_valid(symbol)
        status = "صحيح ✅" if is_valid else "غير صحيح ❌"
        print(f"   {symbol}: {status}")
    
    # 8. اختبار الدوال السريعة
    print("\n⚡ اختبار الدوال السريعة:")
    
    # جلب سعر سريع
    quick_price = get_crypto_price('BTC')
    if quick_price:
        print(f"   السعر السريع لـ BTC: ${quick_price:,.2f}")
    
    # جلب بيانات سريعة
    quick_data = get_crypto_data('ETH', '1mo')
    if quick_data is not None:
        print(f"   البيانات السريعة لـ ETH: {len(quick_data)} يوم")
    
    print("\n🎉 انتهى الاختبار! جميع المميزات تعمل مجاناً بدون API keys!")

def test_data_analysis():
    """اختبار تحليل البيانات المتقدم"""
    
    print("\n" + "="*50)
    print("📊 تحليل متقدم للبيانات")
    print("="*50)
    
    api = BinanceAPI()
    
    # جلب بيانات تفصيلية لـ Bitcoin
    data = api.get_market_data_by_period('BTC', '3mo')
    
    if data is not None:
        print(f"\n📈 تحليل Bitcoin للـ 3 أشهر الماضية:")
        print(f"   عدد نقاط البيانات: {len(data)}")
        print(f"   أعلى سعر: ${data['High'].max():,.2f}")
        print(f"   أقل سعر: ${data['Low'].min():,.2f}")
        print(f"   السعر الحالي: ${data['Close'].iloc[-1]:,.2f}")
        
        # حساب التغيير
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
        price_change_percent = (price_change / data['Close'].iloc[0]) * 100
        
        print(f"   التغيير في 3 أشهر: ${price_change:,.2f} ({price_change_percent:+.1f}%)")
        
        # حساب التقلبات
        daily_returns = data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (365**0.5) * 100  # تقلبات سنوية
        
        print(f"   التقلبات السنوية: {volatility:.1f}%")
        
        # أيام الارتفاع والانخفاض
        up_days = (daily_returns > 0).sum()
        down_days = (daily_returns < 0).sum()
        
        print(f"   أيام الارتفاع: {up_days}")
        print(f"   أيام الانخفاض: {down_days}")
        print(f"   نسبة الارتفاع: {up_days/(up_days+down_days)*100:.1f}%")
        
        # حساب مستويات الدعم والمقاومة
        recent_data = data.tail(30)  # آخر 30 يوم
        support_level = recent_data['Low'].min()
        resistance_level = recent_data['High'].max()
        
        print(f"   مستوى الدعم (30 يوم): ${support_level:,.2f}")
        print(f"   مستوى المقاومة (30 يوم): ${resistance_level:,.2f}")
        
        # عرض عينة من البيانات
        print(f"\n📋 آخر 5 أيام من البيانات:")
        print(data.tail().round(2))
    
    else:
        print("فشل في جلب البيانات للتحليل")

if __name__ == "__main__":
    try:
        # اختبار أساسي
        test_binance_api()
        
        # اختبار متقدم
        test_data_analysis()
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()