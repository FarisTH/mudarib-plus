"""
تطبيق اختبار بسيط لـ Streamlit
Simple Streamlit Test App
"""

import streamlit as st

# تحقق من وجود streamlit
print("Streamlit is working!")

# إعداد الصفحة
st.set_page_config(
    page_title="Test App",
    page_icon="🚀",
    layout="wide"
)

# العنوان
st.title("🚀 اختبار التطبيق - Test App")

# رسالة ترحيب
st.success("✅ Streamlit يعمل بشكل صحيح!")

# اختبار المكونات الأساسية
st.header("🔧 اختبار المكونات:")

# نص بسيط
st.write("📝 النص يعمل!")

# أرقام
st.metric("📊 اختبار المقاييس", "100", "10")

# أزرار
if st.button("🔘 اضغط هنا"):
    st.balloons()
    st.success("الزر يعمل!")

# شريط جانبي
st.sidebar.title("📋 القائمة الجانبية")
st.sidebar.write("القائمة الجانبية تعمل!")

# معلومات النظام
st.header("💻 معلومات النظام:")

import sys
import pandas as pd

info_data = {
    "المعلومة": ["Python Version", "Streamlit Version", "Working Directory"],
    "القيمة": [sys.version.split()[0], st.__version__, "localhost:8502"]
}

df = pd.DataFrame(info_data)
st.table(df)

# اختبار الرسم البياني البسيط
st.header("📈 اختبار الرسم البياني:")

import pandas as pd
import numpy as np

# بيانات وهمية
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

st.line_chart(chart_data)

# رسالة نهاية
st.success("🎉 جميع الاختبارات نجحت!")