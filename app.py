import streamlit as st
from moodbot import MoodBot
import time

@st.cache_resource
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
            response_split = response.split()

            placeholder = st.empty()
            displayed_text = ""
            # Typewriter effect.
            for word in response_split:
                if 'span' in word:
                    displayed_text += word
                    displayed_text += ' '
                    continue

                for char in word:
                    displayed_text += char
                    placeholder.markdown(displayed_text, unsafe_allow_html=True)
                    time.sleep(0.03)

                displayed_text += ' '

    else:
        with st.chat_message('ai'):
            st.write("Hello there! The name's :orange[Moody].")
        
