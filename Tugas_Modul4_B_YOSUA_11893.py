import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances

def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])
    data[f"{model_name}_Cluster"] = clusters

    if model_name == "KMeans_model":
        new_cluster = model.predict(new_point[features])[0]
    else:
        distances = pairwise_distances(new_point[features], data[features])
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    fig = px.scatter_3d(data, x='Avg_Credit_Limit', y='Total_Credit_Cards', z='Total_visits_online',
                        color=f"{model_name}_Cluster", title=title, color_continuous_scale=color_scale)

    fig.add_trace(
        go.Scatter3d(
            x=new_point['Avg_Credit_Limit'],
            y=new_point['Total_Credit_Cards'],
            z=new_point['Total_visits_online'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='New Point'
        )
    )
    return fig, new_cluster

st.set_page_config(
    page_title="11893 - Unsupervised Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning - YOSUA</h1>", unsafe_allow_html=True)  # YYYYY diisi dengan nama panggilan
    st.dataframe(input_data)

# model_directory = r'C:\Pembelajaran Mesin dan Pembelajaran Mendalam\Tugas4_B_11893'
# model_path = {
#     "AGG_model": os.path.join(model_directory, "AGG_model.pkl"),
#     "KMeans_model": os.path.join(model_directory, "KMeans_model.pkl"),
#     "DBSCAN_model": os.path.join(model_directory, "DBSCAN_model.pkl"),
# }

# models = {}
# for model_name, path in model_path.items():
#     if os.path.exists(path):
#         with open(path, 'rb') as f:
#             models[model_name] = pickle.load(f)
#     else:
#         st.write(f"Model {model_name} tidak ditemukan di path : {path}")

model_path = {
    "AGG_model": "AGG_model.pkl",
    "KMeans_model": "KMeans_model.pkl",
    "DBSCAN_model": "DBSCAN_model.pkl",
}

models = {}
for model_name, path in model_path.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            models[model_name] = pickle.load(f)
    else:
        st.write(f"Model {model_name} tidak ditemukan di path : {path}")

avg_CL = st.sidebar.number_input("Average Credit Limit", 0, 100000)
sum_CC = st.sidebar.number_input("Total Credit Cards", 0, 80)
sum_VO = st.sidebar.number_input("Total Visits Online", 0, 16)

if st.sidebar.button("Prediksi !"):
    features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_online']

    new_point = pd.DataFrame({
        'Avg_Credit_Limit': [avg_CL],
        'Total_Credit_Cards': [sum_CC],
        'Total_visits_online': [sum_VO]
    })

    cluster_method = [
        ("KMeans_model", models["KMeans_model"], "KMeans Clustering", px.colors.sequential.Cividis),
        ("AGG_model", models["AGG_model"], "Agglomerative Clustering", px.colors.sequential.Mint),
        ("DBSCAN_model", models["DBSCAN_model"], "DBSCAN Clustering", px.colors.sequential.Plasma)
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (model_name, model, title, color_scale) in enumerate(cluster_method):
        fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)
        with cols[i]:
            st.plotly_chart(fig)
            st.markdown(f"<p style='text-align: center;'>Titik Data yang baru masuk ke dalam cluster : {new_cluster}</p>", unsafe_allow_html=True)
