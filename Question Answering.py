from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
contexts = [
    "Machine learning is a subset of artificial intelligence",
    "NLP is used to process human language",
    "Python is a popular programming language"
]

answers = [
    "ML is part of AI",
    "NLP processes text",
    "Python is widely used"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(contexts)

def answer_question(q):
    q_vec = vectorizer.transform([q])
    sim = cosine_similarity(q_vec, X)
    idx = sim.argmax()
    print("Answer:", answers[idx])

while True:
    q = input("\nAsk: ")
    if q == "exit":
        break
    answer_question(q)
