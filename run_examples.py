"""
Runs example inputs through the model and saves the outputs to
examples_output.txt (sample inputs/outputs required by the task).
"""

from model_qa import SemanticFAQBot

EXAMPLES = [
    "When is the deadline for paying college fees?",
    "How do I see how many classes I have attended?",
    "What percentage of attendance do I need to write exams?",
    "Where can I get a letter proving I am a student here?",
    "I forgot my portal login password",
    "What time does the library close?",
    "What is the weather today?",  # off-topic -> should fall back
]


def main():
    bot = SemanticFAQBot()
    lines = []
    for q in EXAMPLES:
        answer, score, matched = bot.get_answer(q)
        block = (
            f"INPUT : {q}\n"
            f"OUTPUT: {answer}\n"
            f"MATCHED FAQ: {matched}\n"
            f"SCORE : {score:.2f}\n"
        )
        print(block)
        lines.append(block)
    with open("examples_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved to examples_output.txt")


if __name__ == "__main__":
    main()
