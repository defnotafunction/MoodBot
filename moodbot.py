import random
import json
import pyttsx3 as tts
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


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



def quote_data() -> list[dict]:
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

quote_data = quote_data()

all_tags = {
    "sad": [
        "lies", "heartbreak", "unhappy-marriage", "lost", 
        "regrets", "death", "apathy", "indifference", "lack-of-love", 
        "lack-of-friendship"
    ],
    "mad": [
        "failure", "hate", "lying"
    ],
    "happy": [
        "miracles", "beatles", "simplicity", "yourself", "be-yourself", 
        "romance", "fairy-tales", "integrity", "faith", "open-mind", 
        "choices", "better-life-empathy", "chocolate", "comedy", 
        "inspiration", "fantasy", "wisdom", "happiness", "dreams", 
        "imagination", "courage", "miracle", "sun", "planning"
    ],
    "neutral": [
        "edison", "deep-thoughts", "knowledge", "live-death-love", 
        "books", "dumbledore", "mind", "humor", "read", "opposite", 
        "drug", "grown-ups", "abilities", "jane-austen", "activism", 
        "friends", "romantic", "mistakes", "age", "tea", "plans", 
        "library", "philosophy", "peace", "fairytales", "contentment", 
        "success", "children", "god", "understanding", "growing-up", 
        "girls", "seuss", "thought", "food", "classic", "wander", 
        "change", "connection", "readers", "christianity", "difficult", 
        "adulthood", "attributed", "good", "novelist-quotes", "marriage", 
        "music", "insanity", "adventure", "alcohol", "obvious", 
        "troubles", "aliteracy", "elizabeth-bennet", "journey", 
        "writing", "friendship", "women", "the-hunger-games", "simile", 
        "literature", "inspirational", "dreaming", "world", "quest", "misattributed-eleanor-roosevelt", 
        "misattributed-to-mother-teresa", "misattributed-mark-twain", 
        "misattributed-to-einstein", "misattributed-to-c-s-lewis"
    ]
}

def tts_talk(text):
    engine = tts.init()
    engine.say(text)
    engine.runAndWait()
    del engine

def get_quote_from_mood(mood):
    tags_match_moods = all_tags[mood]
    quotes_with_tags = [quote for quote in quote_data if any(tag in tags_match_moods for tag in quote['tags'])]
    quote = random.choice(quotes_with_tags)
    print("#".join(quote['tags']))
    return f"And I quote, {quote['quote']} by {quote['author']}"

class MoodBot:
    def __init__(self, name):
        with open('newtrainingdata.json') as data:
            self.training_data = json.load(data)

        self.name = name

        with open('responses.json') as responses:
            self.responses = json.load(responses)
        
       
        
        self.model = make_pipeline(
                    TfidfVectorizer(
                        ngram_range=(1,3),
                        stop_words='english'
                        ),
                    MultinomialNB()
        )
        self.dataset = self.get_dataset()
        training_data, testing_data, training_labels, testing_labels = train_test_split(self.dataset[0], self.dataset[1], random_state=1, test_size=.3)
        self.model.fit(training_data, training_labels)
        print(self.model.score(testing_data, testing_labels))

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

    def display_guess_mood(self, mood):
        mood_word = mood_key.get(mood)
        
        if 'question' in mood_word:
            print(f"{self.name}: I detect that you're asking a {mood_word}")
            return
        
        response = random.choice(self.responses.get(mood_word))
        quote = get_quote_from_mood(mood_word)

        print(f'{self.name} (Sensing {mood_word}): {response} !\nQuote:\n {quote}')
        return f"I'm Sensing that you're {mood_word}, {response}, {quote}"

    def talk(self):
        user_input = input('Talk: ')
        guess = self.guess_mood(user_input)
        print('')
        if guess is None:
            print('Thank you.')
        else:
            tts_talk(self.display_guess_mood(guess))

Moody = MoodBot('the rotting carcass of andrew'.title())


while True:
    Moody.talk()
    print('--------------------------')