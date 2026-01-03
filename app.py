import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="instacart dashboard", layout="wide")

df = pd.read_csv('dashboard_sample.csv')

st.title("instacart customer behavior dashboard")
st.markdown("---")


col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="task a: accuracy", value="90.66%")
with col2:
    st.metric(label="task b: mae", value="7.0213 days")
with col3:
    st.metric(label="model status", value="robust")

st.markdown("---")


st.subheader("data exploration summary")
st.write(df.describe())


tab1, tab2 = st.tabs(["outlier analysis", "feature distribution"])
with tab1:
    if 'user_days_since_last_order' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df['user_days_since_last_order'], color='skyblue', ax=ax1)
        st.pyplot(fig1)

with tab2:
    selected = st.selectbox("select feature", df.columns)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(df[selected], kde=True, ax=ax2, color='salmon')
    st.pyplot(fig2)

st.caption("final project - ds230")
