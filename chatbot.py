import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab') # Added to fix the LookupError

def chatbot():
    print("Chatbot (type 'exit' to stop)\n")

    while True:
        user = input("You: ").lower()

        if user == "exit":
            print("Bot: Goodbye!")
            break

        tokens = word_tokenize(user)

        if "hello" in tokens:
            print("Bot: Hi there!")
        elif "name" in tokens:
            print("Bot: I am an NLP chatbot.")
        elif "course" in tokens:
            print("Bot: This is NLP course.")
        elif "thanks" in tokens:
            print("Bot: You're welcome!")
        else:
            print("Bot: I don't understand.")

chatbot()
