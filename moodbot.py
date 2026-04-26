import random
import json
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

goemotions_path = os.path.join(BASE_DIR, 'botdata', 'goemotions_1.csv')
mood_to_goemotions_path = os.path.join(BASE_DIR, 'botdata', 'mood_to_goemotions.json')
mood_to_tags_path = os.path.join(BASE_DIR, 'botdata', 'mood_to_tags.json')
new_training_data_path = os.path.join(BASE_DIR, 'botdata', 'new_training_data.json')
responses_path = os.path.join(BASE_DIR, 'botdata', 'motivational_responses.json')
phrases_path = os.path.join(BASE_DIR, 'botdata', 'phrase_variations.json')

MOOD_KEY = {
    0: 'happy',
    1: 'mad',
    2: 'neutral',
    3: 'sad',
    4: 'happy question',
    5: 'mad question',
    6: 'neutral question',
    7: 'sad question'
}

MOOD_COLORS = {
    'happy': '#32CD32',
    'mad': '#FF6347',
    'sad': '#6495ED',
    'neutral': "#8DA0B3"
}

def create_overall_mood_column(row):
    mood_to_goemotions = json.load(open(mood_to_goemotions_path))
    if any(row[col] == 1 for col in mood_to_goemotions['happy']):
        return 0
    if any(row[col] == 1 for col in mood_to_goemotions['mad']):
        return 1
    if any(row[col] == 1 for col in mood_to_goemotions['neutral']):
        return 2
    return 3

emotions = pd.read_csv(goemotions_path)
emotions['overall_mood'] = emotions.apply(create_overall_mood_column, axis=1)


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


def get_quote_from_mood(mood):
    tags_match_moods = all_tags[mood]
    quotes_with_tags = [quote 
                        for quote in quote_data 
                        if any(tag in tags_match_moods for tag in quote['tags'])
                        ]
    quote = random.choice(quotes_with_tags)

    return f"{quote['quote']} - {quote['author']}"

class MoodBot:
    def __init__(self, name, dataset_to_use='original'):
        self.training_data = json.load(open(new_training_data_path))
        self.name = name
        self.responses = json.load(open(responses_path))

        self.model = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        ngram_range=(1,3),  
                        stop_words='english',
                        max_features=5000
                        )),
                    ("lgstic", LogisticRegression(class_weight='balanced') if 'reddit' in dataset_to_use.lower() else MultinomialNB()
                    )
            ]
        ) 
        self.dataset_to_use = dataset_to_use
        self.dataset = self.get_dataset()
        self.phrases = json.load(open(phrases_path))
        self.model.fit(self.dataset[0], self.dataset[1])
        
    def get_dataset(self):
        samples = []
        labels = []

        if 'reddit' in self.dataset_to_use.lower():
            for row in emotions.itertuples():
                sentence = row.text
                label = row.overall_mood
                samples.append(sentence)
                labels.append(label)
                
        elif 'original' in self.dataset_to_use.lower():
            for sentence, label in self.training_data.items():
                samples.append(sentence)
                labels.append(label)
            
        return samples, labels
    
    def guess_mood(self, user_input):
        prediction = self.model.predict([user_input])[0]

        if '?' in user_input:
            return prediction + 4  # Mood questions are greater than their original moods by 4 in mood key
    
        return prediction

    def get_response(self, mood, html_form=True):
        mood_word = MOOD_KEY.get(mood)
        
        if 'question' in mood_word:
            motivational_response = random.choice(self.responses.get(mood_word.split()[0])).lower()
            
            if html_form:
                return f"I believe you're asking a {mood_word}, {motivational_response}."
            return f"<p>I believe you're asking a <span style='color:{MOOD_COLORS.get(mood_word)}'>{mood_word}</span>, {motivational_response}.</p>"
        

        phrase = random.choice(self.phrases.get(mood_word))
        motivational_response = random.choice(self.responses.get(mood_word)).lower()
        quote = get_quote_from_mood(mood_word)
        
        if html_form:
            response = f"<p>{phrase}. I'm sensing that you're <span style='color:{MOOD_COLORS.get(mood_word)}'>{mood_word}</span>, {motivational_response}. \n{quote}</p>"
        else:
            response = f"{phrase}. I'm sensing that you're {mood_word}, {motivational_response}. \n{quote}"
        
        return response

torch.manual_seed(1)

class MoodANN:
    def __init__(self):
        self.training_data = json.load(open(new_training_data_path))
        self.responses = json.load(open(responses_path))
        self.phrases = json.load(open(phrases_path))

        self.vectorizer = TfidfVectorizer(
                        ngram_range=(1,3),  
                        stop_words='english',
                        max_features=10000
                        )
        
        self.dataset = self.get_dataset()
        
    
        X_train, X_test, y_train, y_test = train_test_split(self.dataset[0], self.dataset[1], test_size=0.3, random_state=1)

        self.vectorizer.fit_transform(self.dataset[0], self.dataset[1])
        input_dim = len(self.vectorizer.get_feature_names_out())


        X_train = self.vectorizer.transform(X_train)
        X_test  = self.vectorizer.transform(X_test)

        self.X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        self.X_test  = torch.tensor(X_test.toarray(), dtype=torch.float32)

        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test  = torch.tensor(y_test, dtype=torch.long)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        #self.train_model()
        self.model.load_state_dict(torch.load("models/mood_ann.pth"))

    def get_dataset(self):
        samples = []
        labels = []

        for row in emotions.itertuples():
            sentence = row.text
            label = row.overall_mood
            samples.append(sentence)
            labels.append(label)
        
        return samples, labels

    def train_model(self):
        num_epochs = 1000
        self.model.train()

        for i in range(num_epochs):
            predictions = self.model(self.X_train)
            loss = self.criterion(predictions, self.y_train)

            loss.backward()
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % 100 == 0:
                print(f'Epoch {i}: Loss - {loss.item()}')

        torch.save(self.model.state_dict(), 'models/mood_ann.pth')

    def guess_mood(self, text):
        # Evaluation Mode.
        self.model.eval()
        x = self.vectorizer.transform([text])
        x = torch.tensor(x.toarray(), dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(x)
            pred = torch.argmax(outputs, dim=1).item()

        return pred

    def get_response(self, mood, html_form=True):  
        mood_word = MOOD_KEY.get(mood)
        
        if 'question' in mood_word:
            motivational_response = random.choice(self.responses.get(mood_word.split()[0])).lower()
            
            if html_form:
                return f"I believe you're asking a {mood_word}, {motivational_response}."
            return f"<p>I believe you're asking a <span style='color:{MOOD_COLORS.get(mood_word)}'>{mood_word}</span>, {motivational_response}.</p>"
        

        phrase = random.choice(self.phrases.get(mood_word))
        motivational_response = random.choice(self.responses.get(mood_word)).lower()
        quote = get_quote_from_mood(mood_word)
        
        if html_form:
            response = f"<p>{phrase}. I'm sensing that you're <span style='color:{MOOD_COLORS.get(mood_word)}'>{mood_word}</span>, {motivational_response}. \n{quote}</p>"
        else:
            response = f"{phrase}. I'm sensing that you're {mood_word}, {motivational_response}. \n{quote}"
        
        return response
        

    

md = MoodANN()