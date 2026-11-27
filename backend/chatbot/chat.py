"""Chat with the bot in the terminal.

Run from the project folder:
    .\\venv\\Scripts\\python.exe -m backend.chatbot.chat

Type a question and press Enter. Type exit to stop.
"""

import logging

from backend.chatbot.graph import build_graph
from backend.chatbot.llm import LLMConfigError


def main():
    # Show our own log lines ("Groq ready", fallback warnings), but hide
    # the HTTP request lines that the libraries print.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        app = build_graph()
    except LLMConfigError as error:
        print(f"Can't start: {error}")
        return

    print("Ask me anything. Type exit to stop.\n")
    history = []  # the whole conversation, so follow-up questions make sense

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+Z / Ctrl+C
            print()
            break

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        try:
            result = app.invoke({"messages": history + [("user", question)]})
        except Exception as error:
            print(f"Bot: Sorry, something went wrong: {error}\n")
            continue

        history = result["messages"]
        print(f"Bot: {history[-1].content}\n")

    print("Bye!")


if __name__ == "__main__":
    main()
