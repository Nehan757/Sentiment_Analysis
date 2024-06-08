import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import pickle

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        preprocessed_sentences.append(' '.join(words))
    return preprocessed_sentences

# Improved subtheme extraction function
def extract_subthemes(text):
    doc = nlp(text)
    subthemes = [chunk.text for chunk in doc.noun_chunks]
    return subthemes

sentiment_pipeline = pipeline("sentiment-analysis")

# Sentiment analysis function
def analyze_sentiments(sentences):
    sentiments = [(sentence, sentiment_pipeline(sentence)[0]['label']) for sentence in sentences]
    return sentiments

# Improved matching of subthemes with sentiments
def get_subtheme_sentiments(review):
    preprocessed_sentences = preprocess_text(review)
    subthemes = extract_subthemes(review)
    sentiments = analyze_sentiments(preprocessed_sentences)

    subtheme_sentiments = {}
    for subtheme in subthemes:
        subtheme_sentiments[subtheme] = 'NEUTRAL'

    for sentence, sentiment in sentiments:
        for subtheme in subthemes:
            if subtheme in sentence or subtheme.lower() in sentence.lower():
                subtheme_sentiments[subtheme] = sentiment

    # Handle pronouns
    resolved_subtheme_sentiments = []
    for subtheme, sentiment in subtheme_sentiments.items():
        if subtheme.lower() in ['it', 'they', 'them']:
            # Find the previous subtheme that is not a pronoun
            for prev_subtheme, prev_sentiment in reversed(resolved_subtheme_sentiments):
                if prev_subtheme.lower() not in ['it', 'they', 'them']:
                    subtheme = prev_subtheme
                    break
        resolved_subtheme_sentiments.append((subtheme, sentiment))

    # Remove duplicates by keeping the first occurrence
    unique_subtheme_sentiments = []
    seen_subthemes = set()
    for subtheme, sentiment in resolved_subtheme_sentiments:
        if subtheme not in seen_subthemes:
            unique_subtheme_sentiments.append((subtheme, sentiment))
            seen_subthemes.add(subtheme)

    return unique_subtheme_sentiments

if __name__ == "__main__":
    # Example review
    review = "One tyre went missing, so there was a delay to get the two tyres fitted. The way garage dealt with it was fantastic."

    # Save the function as a pickle file
    with open('subtheme_sentiment_model.pkl', 'wb') as f:
        pickle.dump(get_subtheme_sentiments, f)
