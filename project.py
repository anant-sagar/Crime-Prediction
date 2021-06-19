import pandas as pd
import pandas_profiling
import streamlit as st
import numpy as np

from streamlit_pandas_profiling import st_profile_report

@st.cache
def load_data(path = "datasets/communities-crime-clean.csv"):
    df = pd.read_csv(path)
    df['highCrime'] = np.where(df['ViolentCrimesPerPop']>0.1, 1, 0)
    cols = ['communityname','state','ViolentCrimesPerPop','fold']
    df.drop(cols,axis=1,inplace=True)
    return df

df =load_data()
pr = df[:100].copy().profile_report()
st_profile_report(pr)