#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
#model = load_model('catboost_final')


# In[2]:


def user_input_features():
    Time = st.sidebar.slider('Time', 0, 10000, 25000)
    V1 = st.sidebar.slider('V1', -27, 0, 1)
    V2 = st.sidebar.slider('V2', -34, 0, 9)
    V3 = st.sidebar.slider('V3', -16, 0, 5)
    V4 = st.sidebar.slider('V4', -5, 0, 11)
    V5 = st.sidebar.slider('V5', -33, 0, 12)
    V6 = st.sidebar.slider('V6', -33, 0, 12)
    V7 = st.sidebar.slider('V7', -33, 0, 12)
    V8 = st.sidebar.slider('V8', -33, 0, 12)
    V9 = st.sidebar.slider('V9', -33, 0, 12)
    V10 = st.sidebar.slider('V10', -33, 0, 12)
    V11= st.sidebar.slider('V11', -33, 0, 12)
    V12= st.sidebar.slider('V12', -33, 0, 12)
    V13= st.sidebar.slider('V13',-33, 0, 12)
    V14= st.sidebar.slider('V14', -33, 0, 12)
    V15= st.sidebar.slider('V15', -33, 0, 12)
    V16= st.sidebar.slider('V16', -33, 0, 12)
    V17 = st.sidebar.slider('V17', -33, 0, 12)
    V18= st.sidebar.slider('V18', -33, 0, 12)
    V19= st.sidebar.slider('V19',-33, 0, 12)
    V20= st.sidebar.slider('V20', -33, 0, 12)
    V21= st.sidebar.slider('V21',-33, 0, 12)
    V22= st.sidebar.slider('V22',-33, 0, 12)
    V23= st.sidebar.slider('V23', -33, 0, 12)
    V24= st.sidebar.slider('V24', -33, 0, 12)
    V25= st.sidebar.slider('V25', -33, 0, 12)
    V26= st.sidebar.slider('V26', -33, 0, 12)
    V27= st.sidebar.slider('V27', -33, 0, 12)
    V28= st.sidebar.slider('V28', -33, 0, 12)
    Amount= st.sidebar.slider('Amount', 0, 3000, 7000)
    
    data = {'Time':Time,
           'V1': V1, 
            'V2': V2, 
           'V3': V3,
           'V4': V4,
           'V5': V5,
           'V6': V6,
           'V7': V7,
           'V8': V8,
           'V9': V9,
           'V10': V10,
           'V11': V11,
           'V12': V12,
           'V13': V13,
           'V14': V14,
           'V15': V15,
           'V16': V16,
           'V17': V17,
           'V18': V18,
           'V19': V19,
           'V20': V20,
           'V21': V21,
           'V22': V22,
           'V23': V23,
           'V24': V24,
           'V25': V25,
           'V26': V26,
           'V27': V27,
           'V28': V28,
           'Amount': Amount
           }
 
    features= pd.DataFrame(data, index=[0])
    return features


# In[3]:


if __name__ == '__main__':
 
    st.write("""
    # Credit Card Fraud Detection
    """)
 
    st.sidebar.header('User Input Parameters')
    df =  user_input_features()
    st.subheader('User Input Parameters')
    st.write(df)
 
    #filename = 'D:/IPSR/PYCARAT/catboost_final.pkl'
    model = load_model('catboost_final')
    pred = model.predict(df)
    st.subheader('1 denotes frauduluent transaction, 0 denotes non-fraudulent transaction')
    st.write(pred)
    


# In[4]:





# In[ ]:




