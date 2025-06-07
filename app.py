

import os

file_path = 'pipe.pkl'

print(f"Current working directory: {os.getcwd()}")
print(f"Does '{file_path}' exist? {os.path.exists(file_path)}")
if os.path.exists(file_path):
    print(f"Size of '{file_path}': {os.path.getsize(file_path)} bytes")

import streamlit as st
import pickle
import sklearn
import pandas as pd

from joblib import load
pipe = pickle.load(open('pipe.pkl','rb'))


teams=[
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities=['Chennai', 'Bangalore', 'Bengaluru', 'Kolkata', 'Cape Town',
       'Indore', 'Dharamsala', 'Ranchi', 'Centurion', 'Delhi',
       'Visakhapatnam', 'Mumbai', 'Jaipur', 'Hyderabad', 'Chandigarh',
       'Johannesburg', 'Ahmedabad', 'Bloemfontein', 'Durban',
       'Port Elizabeth', 'Mohali', 'Raipur', 'Nagpur', 'Pune', 'Cuttack',
       'East London', 'Sharjah', 'Kimberley', 'Abu Dhabi']


st.title('IPL Win Predictor')

col1, col2 = st.columns(2)


with col1:
    batting_team=st.selectbox('Select the Batting team : ',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select the bowling team : ',sorted(teams))

selected_city=st.selectbox('Select host City : ',sorted(cities))

target=st.number_input('Target')

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs completed')
with col5:
    wickets=st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left=target=score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr=(runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team + "-"+str(round(win*100))+"%")
    st.subheader(bowling_team + "-"+str(round(loss*100))+"%")