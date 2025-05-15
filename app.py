import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import sklearn

with open("model.pickle", "rb") as f:
    modelo = pickle.load(f)

car = pd.read_csv("car (1).csv")
st.title("Analisis ColGPA")

tab1, tab2, tab3 = st.tabs(["Tab1", "Tab2", "Tab3"])

with tab1:
    st.header("Analisis Univariado")
    fig, ax = plt.subplots(1,4, figsize=(10,4))


    #colGPA
    ax[0].hist(car["colGPA"])
    ax[0].set_title("Promedio Universidad")

    #sexo
    conteo = car["sexo"].value_counts()
    ax[1].bar(conteo.index, conteo.values)
    ax[1].set_title("Sexo")

    #admin
    conteo = car["business"].value_counts()
    ax[2].bar(conteo.index, conteo.values)
    ax[2].set_title("Estudia Admin")


    #clubs
    conteo = car["clubs"].value_counts()
    ax[3].bar(conteo.index, conteo.values)
    ax[3].set_title("Pertenece a un club")


    fig.tight_layout()

    st.pyplot(fig)

with tab2:
    st.header("Analisis Bivariado")

    fig, ax = plt.subplots(1,3,figsize=(10,4))

    #ColGPA vs Business
    sns.boxplot(data=car, x="business", y="colGPA", ax=ax[0])
    ax[0].set_title("Promedio vs  Admin")


    #ColGPA vs Clubs
    sns.boxplot(data=car, x="clubs", y="colGPA", ax=ax[1])
    ax[1].set_title("Promedio vs Clubs")

    #ColGPAvs Sexo
    sns.boxplot(data=car, x="sexo", y="colGPA", ax=ax[2])
    ax[2].set_title("Promedio vs Sexo")


    fig.tight_layout()

    st.pyplot(fig)

with tab3:
    business = st.selectbox("business", ["SI", "NO"])
    if business == "SI": 
        business = 1
    else:
        business = 0
    
    clubs = st.selectbox("clubs", ["SI", "NO"])
    if clubs == "SI": 
        clubs = 1
    else:
        clubs = 0

    sexo = st.selectbox("sexo", ["Hombre", "Mujer"])
    if sexo == "Hombre":
        sexo = 1
    else:
        sexo = 0

    if st.button("Predecir"):
        pred = modelo.predict(np.array([[business, sexo, clubs]]))
        st.write(f"Su promedio sería {round(pred[0],1)}")



