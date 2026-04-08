import streamlit as st
from moodbot import MoodBot
import time

@st.cache_resource
def get_moodbot(dataset_to_use, key):
    return MoodBot('Moody', dataset_to_use=dataset_to_use)



st.title('Your :orange[Motivational] Mood-Bot')
st.subheader('Predict :orange[mood]. Respond accordingly.')
dataset_to_use = st.selectbox(
    'Datasets',
    ('Original + Naive Bayes (approx. 1700 sentences)', 'Reddit + LogisticRegression (approx. 70000 sentences)')
)

mood_bot = get_moodbot(dataset_to_use, key=dataset_to_use)

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
        
