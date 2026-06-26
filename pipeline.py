# This file exists for backward compatibility (CLI usage: python pipeline.py)
# The actual pipeline logic is in app/pipeline/
import os
from dotenv import load_dotenv
load_dotenv()

from app.pipeline.graph import chat, pipeline

# Interactive Chat Loop
if __name__ == "__main__":
    print("=" * 60)
    print("YojnaSahay - Government Scheme Assistant")
    print("Type 'quit' to exit, 'reset' to start new conversation")
    print("=" * 60)

    history = []

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = []
            print("Conversation reset!")
            continue

        response = chat(user_input, history)
        print(f"\nYojnaSahay: {response}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
