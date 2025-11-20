from mentor_engine.mentor import generate_response

print("💬 WealthPlay Mentor Ready. Type your message.")

while True:
    user = input("\nYou: ")
    if user.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    reply = generate_response(user)
    print("\nMentor:", reply)
