import pandas as pd
import numpy as np
import pickle
import streamlit as st

pickle_in = open('model.pkl', 'rb')
lr = pickle.load(pickle_in)

def welcome():
    return 'Selamat Datang'

def prediction(sepal_length, sepal_width, petal_length, petal_width):

    prediction = lr.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

def main():
    st.title("Aplikasi Prediksi Bunga Iris Algoritma Logistic Regression")
    st.markdown('Oleh : Agung Gunawan (2019230044) | Universitas Darma Persada | UAS Datamining')
    st.write('\n')
    st.markdown('Silakan isi form berikut terlebih dahulu :')
    
    st.write('\n')
    sepal_length = st.number_input("Sepal Length", 0)
    sepal_width = st.number_input("Sepal Width", 0)
    petal_length = st.number_input("Petal Length", 0)
    petal_width = st.number_input("Petal Width", 0)
    result =""
    
    if st.button("PREDIKSI"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    st.success('Hasil Prediksi - {}'.format(result))
    
if __name__=='__main__':
    main()