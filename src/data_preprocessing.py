import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenization
        tokens = text.split()

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)
