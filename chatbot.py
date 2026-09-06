import json
import string
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def download_nltk_resources():
    resource_map = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for lookup_path, package_name in resource_map.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                # If download fails (e.g. no internet), we fall back gracefully
                # later in _preprocess() and the stopwords/lemmatizer setup.
                pass


class FAQChatbot:
    def __init__(self, faq_path="faqs.json", similarity_threshold=0.25):
        download_nltk_resources()

        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = set()

        self.similarity_threshold = similarity_threshold
        self.questions, self.answers = self._load_faqs(faq_path)
        self.processed_questions = [self._preprocess(q) for q in self.questions]

        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    @staticmethod
    def _load_faqs(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: FAQ file '{path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: FAQ file is not valid JSON ({e}).")
            sys.exit(1)

        if not data:
            print("Error: FAQ file is empty.")
            sys.exit(1)

        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]
        return questions, answers

    def _preprocess(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()

        cleaned = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok.strip() and tok not in self.stop_words
        ]
        return " ".join(cleaned) if cleaned else text

    def get_response(self, user_input):
        if not user_input or not user_input.strip():
            return "Please type a question so I can help you.", 0.0

        processed_input = self._preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])

        similarities = cosine_similarity(input_vector, self.question_vectors)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score < self.similarity_threshold:
            return (
                "I'm sorry, I don't have an answer for that. "
                "Could you try rephrasing your question?",
                best_score,
            )

        return self.answers[best_idx], best_score

    def chat(self):
        print("FAQ Chatbot: Hi! Ask me anything (type 'quit' or 'exit' to stop).\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nFAQ Chatbot: Goodbye!")
                break

            if user_input.lower() in ("quit", "exit", "bye"):
                print("FAQ Chatbot: Goodbye!")
                break

            answer, score = self.get_response(user_input)
            print(f"FAQ Chatbot: {answer}  (confidence: {score:.2f})\n")


if __name__ == "__main__":
    bot = FAQChatbot(faq_path="faqs.json", similarity_threshold=0.25)
    bot.chat()