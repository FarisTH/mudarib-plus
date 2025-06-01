"""
مضارب بلس - نسخة محدثة وثابتة
Mudarib Plus - Fixed and Stable Version
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
    initial_sidebar_state="expanded"
)

# ===== العنوان الرئيسي =====
st.title("📈 مضارب بلس - Mudarib Plus")
st.markdown("### تطبيق ذكي للتحليل الفني - Smart Technical Analysis")

# تحقق من حالة المكتبات
if not yf_available:
    st.error("❌ مكتبة yfinance غير متوفرة. يرجى تثبيتها: `pip install yfinance`")
    
if not plotly_available:
    st.warning("⚠️ مكتبة plotly غير متوفرة. سيتم استخدام الرسوم البديلة.")

# ===== الشريط الجانبي =====
st.sidebar.header("⚙️ إعدادات التحليل")

# معلومات حالة النظام
st.sidebar.markdown("### 📊 حالة النظام:")
st.sidebar.success(f"✅ Streamlit v{st.__version__}")
st.sidebar.info(f"✅ Pandas v{pd.__version__}")
st.sidebar.info(f"✅ NumPy v{np.__version__}")

if yf_available:
    st.sidebar.success("✅ YFinance متوفر")
else:
    st.sidebar.error("❌ YFinance غير متوفر")

if plotly_available:
    st.sidebar.success("✅ Plotly متوفر")
else:
    st.sidebar.warning("⚠️ Plotly غير متوفر")

# ===== المحتوى الرئيسي =====
if yf_available:
    # اختيار نوع الأصل
    asset_type = st.sidebar.selectbox(
        "🎯 نوع الأصل",
        options=["الأسهم", "العملات الرقمية"],
        index=0
    )

    if asset_type == "الأسهم":
        symbol = st.sidebar.selectbox(
            "📈 اختر السهم",
            options=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"],
            index=0
        )
    else:
        symbol = st.sidebar.selectbox(
            "₿ اختر العملة الرقمية",
            options=["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD"],
            index=0
        )

    period = st.sidebar.selectbox(
        "📅 الفترة الزمنية",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=1
    )

    # ===== جلب البيانات =====
    @st.cache_data(ttl=300)
    def get_data(symbol, period):
        """جلب البيانات من yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"خطأ في جلب البيانات: {str(e)}")
            return None

    # زر جلب البيانات
    if st.sidebar.button("🔄 جلب البيانات"):
        with st.spinner(f"🔄 جاري جلب بيانات {symbol}..."):
            data = get_data(symbol, period)
        
        if data is not None and not data.empty:
            st.success(f"✅ تم جلب {len(data)} صف من البيانات بنجاح!")
            
            # ===== المعلومات الأساسية =====
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            with col1:
                st.metric(
                    label="💰 السعر الحالي",
                    value=f"${current_price:.2f}",
                    delta=f"{change_percent:+.2f}%"
                )
            
            with col2:
                st.metric(
                    label="📊 أعلى سعر",
                    value=f"${data['High'].max():.2f}"
                )
            
            with col3:
                st.metric(
                    label="📉 أقل سعر",
                    value=f"${data['Low'].min():.2f}"
                )
            
            with col4:
                st.metric(
                    label="📈 متوسط الحجم",
                    value=f"{data['Volume'].mean():,.0f}"
                )
            
            # ===== الرسم البياني =====
            st.subheader("📊 الرسم البياني")
            
            if plotly_available:
                # رسم بياني متقدم مع Plotly
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ))
                
                fig.update_layout(
                    title=f"📈 {symbol} - تحليل الأسعار",
                    xaxis_title="التاريخ",
                    yaxis_title="السعر ($)",
                    template="plotly_dark",
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # رسم بياني بسيط مع Streamlit
                st.line_chart(data['Close'])
            
            # ===== المؤشرات البسيطة =====
            st.subheader("📈 المؤشرات الفنية")
            
            # حساب المتوسطات المتحركة
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            if plotly_available:
                # رسم بياني للمتوسطات مع Plotly
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='السعر',
                    line=dict(color='white', width=2)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MA20'],
                    mode='lines',
                    name='المتوسط المتحرك 20',
                    line=dict(color='#00ff88', width=1)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MA50'],
                    mode='lines',
                    name='المتوسط المتحرك 50',
                    line=dict(color='#ff4444', width=1)
                ))
                
                fig2.update_layout(
                    title="📊 السعر مع المتوسطات المتحركة",
                    xaxis_title="التاريخ",
                    yaxis_title="السعر ($)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # رسم بسيط للمتوسطات
                chart_data = pd.DataFrame({
                    'السعر': data['Close'],
                    'MA20': data['MA20'],
                    'MA50': data['MA50']
                })
                st.line_chart(chart_data)
            
            # ===== جدول البيانات =====
            st.subheader("📋 البيانات التفصيلية")
            
            display_data = data.tail(10).copy()
            display_data = display_data.round(2)
            display_data.index = display_data.index.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_data[['Open', 'High', 'Low', 'Close', 'Volume']],
                use_container_width=True
            )
            
            # ===== الإحصائيات =====
            st.subheader("📊 الإحصائيات")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 إحصائيات الأسعار:**")
                stats_df = pd.DataFrame({
                    'المؤشر': ['المتوسط', 'الوسيط', 'الانحراف المعياري', 'أعلى سعر', 'أقل سعر'],
                    'القيمة': [
                        f"${data['Close'].mean():.2f}",
                        f"${data['Close'].median():.2f}",
                        f"${data['Close'].std():.2f}",
                        f"${data['Close'].max():.2f}",
                        f"${data['Close'].min():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            
            with col2:
                st.write("**📊 إحصائيات الحجم:**")
                volume_stats = pd.DataFrame({
                    'المؤشر': ['متوسط الحجم', 'أعلى حجم', 'أقل حجم'],
                    'القيمة': [
                        f"{data['Volume'].mean():,.0f}",
                        f"{data['Volume'].max():,.0f}",
                        f"{data['Volume'].min():,.0f}"
                    ]
                })
                st.dataframe(volume_stats, hide_index=True)
        else:
            st.error("❌ فشل في جلب البيانات. يرجى المحاولة مرة أخرى.")

else:
    # عرض بيانات وهمية إذا لم تكن yfinance متوفرة
    st.warning("⚠️ yfinance غير متوفر. عرض بيانات تجريبية:")
    
    # إنشاء بيانات وهمية
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
    dummy_data = pd.DataFrame({
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    st.line_chart(dummy_data['Close'])
    st.bar_chart(dummy_data['Volume'])

# ===== تذييل الصفحة =====
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📈 مضارب بلس - Mudarib Plus | تطبيق ذكي للتحليل الفني</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== معلومات التشخيص =====
st.sidebar.markdown("---")
st.sidebar.success("✅ التطبيق يعمل!")
st.sidebar.info(f"🕒 الوقت: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.info(f"🌐 URL: localhost:8502")