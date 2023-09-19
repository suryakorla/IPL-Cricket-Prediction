import streamlit as st 
import pandas as pd
import pickle
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

model=pickle.load(open('pipe.pkl','rb'))

teams=['Rajasthan Royals',
 'Royal Challengers Bangalore',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Chennai Super Kings',
 'Gujarat Titans',
 'Lucknow Super Giants',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Mumbai Indians']

City=['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
       'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
       'Bangalore', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur',
       'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein',
       'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town']

st.title("CRICKET WINNER PREDICTION")

col1,col2=st.columns(2)


with col1:
    BattingTeam=st.selectbox('Select the Batting Team',sorted(teams))
    
with col2:
    BowlingTeam=st.selectbox('Select the Bowling Team',sorted(teams))
    
City=st.selectbox('Select the City',sorted(City))


target=st.number_input('target')


col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score ')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets=st.number_input('Wickets Outs')
    
if st.button('Predict Probability'):
    runs_left = target-score
    balls_left = 120-(overs*6)
    wickets_left = 10-wickets
    current_run_rate = score/overs
    required_run_rate = (runs_left*6)/balls_left


input_df=pd.DataFrame({'BattingTeam':[BattingTeam],'BowlingTeam':[BowlingTeam],'City':[City],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'current_run_rate':[current_run_rate],'required_run_rate':[required_run_rate],'target':[target]})  


result=model.predict_proba(input_df)

loss=result[0][0]
win=result[0][1]

st.header(BattingTeam + " - " +str(round(win*100)+ "%"))
st.header(BowlingTeam + " - " +str(round(loss*100)+ "%"))

