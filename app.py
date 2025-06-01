"""
📈 مضارب بلس - Mudarib Plus
تطبيق ذكي للتحليل الفني للأسهم والعملات الرقمية
Smart Technical Analysis App for Stocks & Cryptocurrencies

المطور: Faris TH
GitHub: https://github.com/FarisTH/mudarib-plus
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# التحقق من الاستيراد وعرض رسائل واضحة
try:
    import yfinance as yf
    yf_available = True
except ImportError:
    yf_available = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="مضارب بلس - Mudarib Plus",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/FarisTH/mudarib-plus',
        'Report a bug': 'https://github.com/FarisTH/mudarib-plus/issues',
        'About': """
        # مضارب بلس - Mudarib Plus
        تطبيق ذكي للتحليل الفني للأسهم والعملات الرقمية
        
        **المطور:** Faris TH  
        **المشروع:** https://github.com/FarisTH/mudarib-plus
        
        Made with ❤️ in Saudi Arabia
        """
    }
)

# ===== CSS مخصص للتحسين =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    
    .success-msg {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===== العنوان الرئيسي =====
st.markdown('<h1 class="main-header">📈 مضارب بلس - Mudarib Plus</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">تطبيق ذكي للتحليل الفني - Smart Technical Analysis</p>', unsafe_allow_html=True)

# ===== الشريط الجانبي =====
st.sidebar.markdown("### ⚙️ إعدادات التحليل")

# معلومات حالة النظام
with st.sidebar.expander("📊 حالة النظام", expanded=False):
    st.success(f"✅ Streamlit v{st.__version__}")
    st.info(f"✅ Pandas v{pd.__version__}")
    st.info(f"✅ NumPy v{np.__version__}")
    
    if yf_available:
        st.success("✅ YFinance متوفر")
    else:
        st.error("❌ YFinance غير متوفر")
    
    if plotly_available:
        st.success("✅ Plotly متوفر")
    else:
        st.warning("⚠️ Plotly غير متوفر")

# ===== إعدادات التداول =====
st.sidebar.markdown("---")

# اختيار نوع الأصل
asset_type = st.sidebar.selectbox(
    "🎯 نوع الأصل المالي",
    options=["العملات الرقمية", "الأسهم الأمريكية", "المؤشرات"],
    index=0
)

# قوائم الأصول
crypto_symbols = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD", 
    "Binance Coin": "BNB-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Solana": "SOL-USD",
    "Dogecoin": "DOGE-USD",
    "Polygon": "MATIC-USD",
    "Chainlink": "LINK-USD",
    "Litecoin": "LTC-USD"
}

stock_symbols = {
    "Apple": "AAPL",
    "Microsoft": "MSFT", 
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "Adobe": "ADBE",
    "Intel": "INTC"
}

index_symbols = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil": "CL=F"
}

# اختيار الرمز بناءً على النوع
if asset_type == "العملات الرقمية":
    selected_name = st.sidebar.selectbox("₿ اختر العملة الرقمية", list(crypto_symbols.keys()))
    symbol = crypto_symbols[selected_name]
    asset_emoji = "₿"
elif asset_type == "الأسهم الأمريكية":
    selected_name = st.sidebar.selectbox("📈 اختر السهم", list(stock_symbols.keys()))
    symbol = stock_symbols[selected_name]
    asset_emoji = "📈"
else:
    selected_name = st.sidebar.selectbox("📊 اختر المؤشر", list(index_symbols.keys()))
    symbol = index_symbols[selected_name]
    asset_emoji = "📊"

# اختيار الفترة الزمنية
period_options = {
    "شهر واحد": "1mo",
    "3 أشهر": "3mo", 
    "6 أشهر": "6mo",
    "سنة واحدة": "1y",
    "سنتان": "2y",
    "5 سنوات": "5y"
}

selected_period = st.sidebar.selectbox("📅 الفترة الزمنية", list(period_options.keys()), index=2)
period = period_options[selected_period]

# فترة المتوسط المتحرك
ma_period = st.sidebar.slider("📈 فترة المتوسط المتحرك", 5, 200, 20)
ma_period_long = st.sidebar.slider("📈 المتوسط المتحرك الطويل", 10, 200, 50)

# ===== المحتوى الرئيسي =====
if not yf_available:
    st.error("""
    ❌ **مكتبة yfinance غير متوفرة**
    
    لتشغيل التطبيق محلياً، قم بتثبيت المكتبة:
    ```bash
    pip install yfinance
    ```
    """)
    st.stop()

# ===== دالة جلب البيانات =====
@st.cache_data(ttl=300, show_spinner=False)
def get_data(symbol, period):
    """جلب البيانات من yfinance مع معالجة الأخطاء"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return None, "لا توجد بيانات للرمز المحدد"
            
        # إضافة المؤشرات الفنية
        data['MA_Short'] = data['Close'].rolling(window=ma_period).mean()
        data['MA_Long'] = data['Close'].rolling(window=ma_period_long).mean()
        
        # حساب RSI مبسط
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data, None
        
    except Exception as e:
        return None, f"خطأ في جلب البيانات: {str(e)}"

# ===== زر جلب البيانات =====
st.sidebar.markdown("---")
fetch_button = st.sidebar.button(
    f"🔄 جلب بيانات {selected_name}",
    type="primary",
    use_container_width=True
)

# ===== عرض البيانات =====
if fetch_button or 'data' in st.session_state:
    
    with st.spinner(f"🔄 جاري جلب بيانات {selected_name} لفترة {selected_period}..."):
        data, error = get_data(symbol, period)
    
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    if data is None or data.empty:
        st.error("❌ لم يتم العثور على بيانات للرمز المحدد")
        st.stop()
    
    # حفظ البيانات في session state
    st.session_state.data = data
    st.session_state.symbol = symbol
    st.session_state.selected_name = selected_name
    st.session_state.asset_emoji = asset_emoji
    
    # رسالة نجاح
    st.success(f"✅ تم جلب {len(data)} صف من البيانات بنجاح!")
    
    # ===== المعلومات الأساسية =====
    current_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    change = current_price - previous_price
    change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
    
    # عرض المقاييس
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{asset_emoji} السعر الحالي",
            value=f"${current_price:,.2f}",
            delta=f"{change_percent:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="📊 أعلى سعر",
            value=f"${data['High'].max():,.2f}"
        )
    
    with col3:
        st.metric(
            label="📉 أقل سعر", 
            value=f"${data['Low'].min():,.2f}"
        )
    
    with col4:
        volume_avg = data['Volume'].mean()
        volume_display = f"{volume_avg/1e6:.1f}M" if volume_avg > 1e6 else f"{volume_avg:,.0f}"
        st.metric(
            label="📈 متوسط الحجم",
            value=volume_display
        )
    
    # ===== الرسم البياني الرئيسي =====
    st.markdown("---")
    st.subheader(f"📊 تحليل {selected_name} - {symbol}")
    
    if plotly_available:
        # رسم الشموع اليابانية مع المتوسطات
        fig = go.Figure()
        
        # الشموع اليابانية
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=selected_name,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ))
        
        # المتوسطات المتحركة
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_Short'],
            mode='lines',
            name=f'MA {ma_period}',
            line=dict(color='#00ff88', width=1),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_Long'],
            mode='lines',
            name=f'MA {ma_period_long}',
            line=dict(color='#ff4444', width=1),
            opacity=0.7
        ))
        
        # تحسين التصميم
        fig.update_layout(
            title=f"📈 {selected_name} ({symbol}) - تحليل الأسعار",
            xaxis_title="التاريخ",
            yaxis_title="السعر ($)",
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=0.01
            )
        )
        
        # إضافة خط الدعم والمقاومة
        support_level = data['Low'].min()
        resistance_level = data['High'].max()
        
        fig.add_hline(y=support_level, line_dash="dash", line_color="green", 
                     annotation_text="دعم", annotation_position="bottom right")
        fig.add_hline(y=resistance_level, line_dash="dash", line_color="red",
                     annotation_text="مقاومة", annotation_position="top right")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # رسم RSI
        if 'RSI' in data.columns:
            st.subheader("📊 مؤشر القوة النسبية (RSI)")
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # خطوط 30 و 70
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                            annotation_text="تشبع شراء (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                            annotation_text="تشبع بيع (30)")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray",
                            annotation_text="خط الوسط (50)")
            
            fig_rsi.update_layout(
                title="📊 مؤشر القوة النسبية (RSI)",
                xaxis_title="التاريخ",
                yaxis_title="RSI",
                template="plotly_dark",
                height=300,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    else:
        # رسم بسيط إذا لم تكن Plotly متوفرة
        st.line_chart(data[['Close', 'MA_Short', 'MA_Long']])
    
    # ===== تحليل فني مبسط =====
    st.markdown("---")
    st.subheader("🎯 التحليل الفني")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 إشارات التداول")
        
        # تحليل المتوسطات المتحركة
        current_ma_short = data['MA_Short'].iloc[-1]
        current_ma_long = data['MA_Long'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        
        if current_ma_short > current_ma_long:
            st.success("🟢 **اتجاه صاعد** - المتوسط القصير أعلى من الطويل")
        else:
            st.error("🔴 **اتجاه هابط** - المتوسط القصير أقل من الطويل")
        
        # تحليل RSI
        if current_rsi > 70:
            st.warning("⚠️ **تشبع شراء** - قد يحدث تصحيح")
        elif current_rsi < 30:
            st.info("💡 **تشبع بيع** - فرصة شراء محتملة")
        else:
            st.info("😐 **منطقة معتدلة** - لا إشارات قوية")
    
    with col2:
        st.markdown("### 📊 إحصائيات مهمة")
        
        stats_data = {
            "المؤشر": [
                "التغير اليومي",
                "المدى (أعلى - أقل)",
                "التقلب (انحراف معياري)",
                f"متوسط {ma_period} يوم",
                f"متوسط {ma_period_long} يوم",
                "RSI الحالي"
            ],
            "القيمة": [
                f"{change_percent:+.2f}%",
                f"${data['High'].max() - data['Low'].min():,.2f}",
                f"{data['Close'].pct_change().std() * 100:.2f}%",
                f"${current_ma_short:.2f}",
                f"${current_ma_long:.2f}",
                f"{current_rsi:.1f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    # ===== جدول البيانات التفصيلية =====
    st.markdown("---")
    st.subheader("📋 البيانات التفصيلية (آخر 10 أيام)")
    
    display_data = data.tail(10).copy()
    display_data = display_data.round(2)
    display_data.index = display_data.index.strftime('%Y-%m-%d')
    
    # اختيار الأعمدة للعرض
    columns_to_show = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_Short', 'MA_Long']
    if 'RSI' in display_data.columns:
        columns_to_show.append('RSI')
    
    st.dataframe(
        display_data[columns_to_show],
        use_container_width=True
    )
    
    # ===== تحليل الحجم =====
    if plotly_available:
        st.markdown("---")
        st.subheader("📊 تحليل الحجم")
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='الحجم',
            marker_color='rgba(50, 171, 96, 0.6)'
        ))
        
        fig_volume.update_layout(
            title="📊 حجم التداول",
            xaxis_title="التاريخ",
            yaxis_title="الحجم",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)

else:
    # صفحة الترحيب
    st.markdown("""
    <div class="info-box">
        <h3>🎯 مرحباً بك في مضارب بلس!</h3>
        <p>تطبيق ذكي شامل للتحليل الفني للأسواق المالية</p>
    </div>
    """, unsafe_allow_html=True)
    
    # معرض الميزات
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### ₿ العملات الرقمية
        - Bitcoin, Ethereum, BNB
        - أكثر من 10 عملات مشهورة
        - بيانات فورية ودقيقة
        """)
    
    with col2:
        st.markdown("""
        ### 📈 الأسهم الأمريكية  
        - Apple, Microsoft, Tesla
        - أشهر الشركات التقنية
        - تحليل شامل للأداء
        """)
    
    with col3:
        st.markdown("""
        ### 📊 المؤشرات العالمية
        - S&P 500, NASDAQ
        - الذهب، النفط، الفضة
        - مؤشر الخوف VIX
        """)
    
    # تعليمات الاستخدام
    st.markdown("---")
    st.markdown("""
    ### 🚀 كيفية الاستخدام:
    1. **اختر نوع الأصل** من القائمة الجانبية
    2. **حدد الرمز** الذي تريد تحليله  
    3. **اختر الفترة الزمنية** للتحليل
    4. **اضغط زر "جلب البيانات"** لبدء التحليل
    5. **استمتع بالتحليل الشامل** والرسوم التفاعلية!
    """)

# ===== تذييل الصفحة =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>📈 مضارب بلس - Mudarib Plus</h4>
    <p>تطبيق ذكي للتحليل الفني | Smart Technical Analysis App</p>
    <p>Made with ❤️ using Streamlit | المطور: Faris TH</p>
    <p>
        <a href="https://github.com/FarisTH/mudarib-plus" target="_blank">
            🔗 GitHub Repository
        </a>
    </p>
</div>
""", unsafe_allow_html=True)

# ===== معلومات إضافية في الشريط الجانبي =====
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ معلومات")
st.sidebar.info(f"🕒 آخر تحديث: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.success("✅ التطبيق يعمل بشكل مثالي!")

# إضافة معلومات المطور
with st.sidebar.expander("👨‍💻 عن المطور"):
    st.markdown("""
    **Faris TH**
    
    🔗 [GitHub](https://github.com/FarisTH)  
    📧 مطور سعودي متخصص في التطبيقات المالية
    
    **التقنيات المستخدمة:**
    - Python & Streamlit
    - YFinance API
    - Plotly للرسوم التفاعلية
    - التصميم المتجاوب
    """)

# ===== Easter Egg =====
if st.sidebar.button("🎉 مفاجأة!"):
    st.balloons()
    st.success("🎊 شكراً لاستخدام مضارب بلس! نتمنى لك تداولاً موفقاً!")