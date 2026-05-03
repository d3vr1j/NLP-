import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

text = """Natural Language Processing is a field of AI.
It helps computers understand human language.
It is widely used in chatbots and translation."""

stop_words = set(stopwords.words('english'))

# Word frequency
words = word_tokenize(text.lower())
freq = defaultdict(int)

for w in words:
    if w not in stop_words:
        freq[w] += 1

# Sentence scoring
sentences = sent_tokenize(text)
scores = {}

for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in freq:
            scores[sent] = scores.get(sent, 0) + freq[word]

# Get best sentence
summary = max(scores, key=scores.get)

print("Summary:\n", summary)
