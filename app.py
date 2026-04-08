import streamlit as st
from moodbot import MoodBot

@st.cache_data
def get_moodbot():
    return MoodBot('Moody')

mood_bot = get_moodbot()

st.title('Your :orange[Motivational] Mood-Bot')
st.subheader('Predict :orange[mood]. Respond accordingly.')
user_prompt = st.chat_input("Say something!")

with st.container(border=True, height=500):
    
    if user_prompt:
        guessed_mood = mood_bot.guess_mood(user_prompt)
        response = mood_bot.get_response(guessed_mood)
        
        with st.chat_message('ai'):
            st.write(response)
    else:
        with st.chat_message('ai'):
            st.write("Hello there! The name's Moody.")
        
