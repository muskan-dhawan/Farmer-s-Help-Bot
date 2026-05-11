from bot.bot_logic import chatbot_response
import os
import sys
import io

# Fix encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set dummy DB for testing
os.environ["BOT_DB_PATH"] = "test_farmer.sqlite3"

def main():
    print("Agricultural Bot CLI Test")
    print("------------------------")
    
    user_id = "test_user_1"
    
    # Test Greeting
    print(f"User: hi")
    print(f"Bot: {chatbot_response(user_id, 'hi')}\n")
    
    # Test Location
    print(f"User: I am in Gujarat")
    print(f"Bot: {chatbot_response(user_id, 'I am in Gujarat')}\n")
    
    # Test Problem (Using RAG)
    print(f"User: wheat pest problem")
    print(f"Bot: {chatbot_response(user_id, 'wheat pest problem')}\n")

if __name__ == "__main__":
    main()
