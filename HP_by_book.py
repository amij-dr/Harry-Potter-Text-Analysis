import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Read the book
with open("HP_Books/Harry Potter and the Sorcerer's Stone.txt", "r", encoding="utf-8") as file:
    book_one = file.read()

# Split into chapters
chapters = re.split(r'CHAPTER [a-zA-Z ]+', book_one)[1:]
chapters = [chapter.strip() for chapter in chapters if chapter.strip()]  # Remove empty chapters

# Process each chapter
processed_chapters = []
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    # Stem
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

# Preprocess chapters
processed_chapters = [preprocess_text(chapter) for chapter in chapters]

# Create and apply TF-IDF vectorizer
vectorizer = TfidfVectorizer(lowercase=False)
tfidf_matrix = vectorizer.fit_transform(processed_chapters)
feature_names = vectorizer.get_feature_names_out()

# Calculate total TF-IDF scores across all chapters
total_tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)

# Find top words
n_top_words = 50
sorted_indices = np.argsort(total_tfidf_scores)[::-1]

print("Top 50 Most Important Words in Harry Potter and the Sorcerer's Stone:")
print("--------------------------------------------------------------")
for rank, idx in enumerate(sorted_indices[:n_top_words], 1):
    word = feature_names[idx]
    score = total_tfidf_scores[idx]
    print(f"{rank}. {word} = {score:.4f}")