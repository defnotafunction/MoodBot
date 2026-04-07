import random
import json
import pyttsx3 as tts
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mood_to_tags_path = os.path.join(BASE_DIR, 'mood_to_tags.json')
new_training_data_path = os.path.join(BASE_DIR, 'new_training_data.json')
responses_path = os.path.join(BASE_DIR, 'motivational_responses.json')
phrases_path = os.path.join(BASE_DIR, 'phrase_variations.json')

mood_key = {
    0: 'happy',
    1: 'mad',
    2: 'neutral',
    3: 'sad',
    4: 'happy question',
    5: 'mad question',
    6: 'neutral question',
    7: 'sad question'
}



def get_quote_data() -> list[dict]:
    """
    Scrapes all quotes from quotes.toscrape.com and returns it in a list
    """
    quotes_list = []

    for i in range(1, 11):
        website_page = i
        quotes_request = requests.get(f"https://quotes.toscrape.com/page/{website_page}/")
        quotes_html = BeautifulSoup(quotes_request.content, "html.parser")
        
        for div in quotes_html.select(".quote"):
            div_dict = {}
            div_in_text = div.get_text()

            for author in quotes_html.select(".author"):
                if author.get_text() in div_in_text:
                    div_dict['author'] = author.get_text()

            for quote in quotes_html.select(".text"):
                if quote.get_text() in div_in_text:
                    div_dict['quote'] = quote.get_text()

            tags_list1 = [tag.get_text().split('\n') for tag in quotes_html.select(".tags") if tag.get_text() in div_in_text]
            tags_list2 = [tag for lst in tags_list1 for idx, tag in enumerate(lst) if idx >= 3 and idx < len(lst)-1]
            div_dict['tags'] = tags_list2
            
            quotes_list.append(div_dict)

    return quotes_list

quote_data = get_quote_data()
all_tags = json.load(open(mood_to_tags_path))

def tts_talk(text):
    engine = tts.init()
    engine.setProperty('rate', 250)
    engine.say(text)
    engine.runAndWait()
    del engine

def get_quote_from_mood(mood):
    tags_match_moods = all_tags[mood]
    quotes_with_tags = [quote 
                        for quote in quote_data 
                        if any(tag in tags_match_moods for tag in quote['tags'])
                        ]
    quote = random.choice(quotes_with_tags)

    return f"{quote['quote']} - {quote['author']}"

class MoodBot:
    def __init__(self, name, model_to_use='naive'):
        self.training_data = json.load(open(new_training_data_path))
        self.name = name
        self.responses = json.load(open(responses_path))
        
        self.model = make_pipeline(
                    TfidfVectorizer(
                        ngram_range=(2,3), 
                        stop_words='english'
                        ),
                    MultinomialNB()
        )
        self.dataset = self.get_dataset()
        self.phrases = json.load(open(phrases_path))

        #training_data, testing_data, training_labels, testing_labels = train_test_split(self.dataset[0], self.dataset[1], random_state=1, test_size=.3)
        self.model.fit(self.dataset[0], self.dataset[1])

    def get_dataset(self):
        samples = []
        labels = []
        
        for sentence, label in self.training_data.items():
            samples.append(sentence)
            labels.append(label)
        
        return samples, labels
    
    def guess_mood(self, user_input):
        prediction = self.model.predict([user_input])[0]

        if '?' in user_input:
            return prediction + 4
    
        return prediction

    def get_response(self, mood):
        mood_word = mood_key.get(mood)
        

        if 'question' in mood_word:
            motivational_response = random.choice(self.responses.get(mood_word.split()[0])).lower()
            return f"I believe you're asking a {mood_word}, {motivational_response}."
        
        phrase = random.choice(self.phrases.get(mood_word))
        motivational_response = random.choice(self.responses.get(mood_word)).lower()
        quote = get_quote_from_mood(mood_word)
        
        response = f"{phrase}. I'm sensing that you're {mood_word}, {motivational_response}. \n{quote}"

        return response

    def speak(self, text):
        tts_talk(text)



Moody = MoodBot('Moody')


