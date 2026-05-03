from googletrans import Translator

translator = Translator()

text = input("Enter text: ")

translated = translator.translate(text, dest='fr')  # French

print("Translated:", translated.text)
