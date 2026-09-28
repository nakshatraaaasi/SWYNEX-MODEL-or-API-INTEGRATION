"""
Student FAQ Q&A - Model Integration (Task 2)
--------------------------------------------
Integrates a pretrained open-source model (sentence-transformers
'all-MiniLM-L6-v2') into the Student FAQ prototype from Task 1.

The model converts questions into semantic embeddings, so queries with
different wording (e.g. "fees kab tak bharni hai") can still match the
right FAQ. No API key is required - the model runs locally.

Run: python model_qa.py
"""

import csv
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.40  # below this score -> "I don't know"


def load_faq(path="faq_data.csv"):
    questions, answers = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append(row["question"])
            answers.append(row["answer"])
    return questions, answers


class SemanticFAQBot:
    def __init__(self, faq_path="faq_data.csv"):
        self.questions, self.answers = load_faq(faq_path)
        self.model = SentenceTransformer(MODEL_NAME)
        self.q_embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def get_answer(self, query):
        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.q_embeddings)[0]
        best = int(scores.argmax())
        score = float(scores[best])
        if score < THRESHOLD:
            return "I don't know this yet. Please contact the college office.", score, None
        return self.answers[best], score, self.questions[best]


def main():
    bot = SemanticFAQBot()
    print("Semantic FAQ Bot ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break
        answer, score, matched = bot.get_answer(query)
        print(f"Bot: {answer}\n     (score: {score:.2f}, matched: {matched})\n")


if __name__ == "__main__":
    main()
