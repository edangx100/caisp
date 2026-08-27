import re
import random
import time

# Define patterns and responses
patterns_responses = [
    (r'\b(hi|hello|hey)\b', ['Hello!', 'Hi there!', 'Hey! How can I help you?']),
    (r'\bhow are you\b', ['I\'m doing well, thanks!', 'I\'m just a program, but I\'m functioning perfectly!']),
    (r'\bwhat is your name\b', ['I\'m SimpleBot, a rule-based chatbot.', 'You can call me SimpleBot!']),
    (r'\btime\b', [f"The current time is {time.strftime('%H:%M:%S')}"]),
    (r'\bweather\b', ['I can\'t check the weather yet, that would require an API integration.']),
    (r'\bjoke\b', ['Why don\'t scientists trust atoms? Because they make up everything!', 
                  'What do you call a fake noodle? An impasta!', 
                  'Why did the scarecrow win an award? Because he was outstanding in his field!']),
    (r'\bthank you\b', ['You\'re welcome!', 'Happy to help!', 'No problem!']),
    (r'\bbye\b', ['Goodbye!', 'See you later!', 'Take care!'])
]

# Initialize context dictionary to store conversation state
context = {
    'user_name': None,
    'last_topic': None,
    'questions_asked': 0
}

# Define the help message
help_message = "I can respond to greetings, tell jokes, show the time, and answer basic questions. Try asking me 'what is your name' or 'tell me a joke'!"

# Display welcome message
print("="*50)
print("Enhanced Rule-Based Chatbot")
print("Type 'bye' to exit the conversation")
print("Type 'help' for assistance")
print("="*50)

# Main chatbot loop
while True:
    # Get user input
    user_input = input("\nYou: ")

    # Update context
    context['questions_asked'] += 1

    # Convert to lowercase for matching
    user_input_lower = user_input.lower()

    # Exit condition
    if user_input_lower == "bye":
        # Personalize goodbye if we know the name
        if context['user_name']:
            print(f"\nChatbot: Goodbye, {context['user_name']}! Have a great day!")
        else:
            print("\nChatbot: Goodbye! Have a great day!")
        break

    # Help command
    if user_input_lower == "help":
        print(f"\nChatbot: {help_message}")
        continue

    # Check for name introduction
    if context['user_name'] is None and 'my name is' in user_input_lower:
        name_match = re.search(r'my name is (\w+)', user_input_lower)
        if name_match:
            context['user_name'] = name_match.group(1).capitalize()
            print(f"\nChatbot: Nice to meet you, {context['user_name']}! How can I help you today?")
            continue

    # Default response
    response = "I'm not sure I understand. Type 'help' for assistance."

    # Try to match patterns
    for pattern, responses in patterns_responses:
        if re.search(pattern, user_input_lower):
            context['last_topic'] = pattern
            response = random.choice(responses)
            break

    # Display the response
    print(f"\nChatbot: {response}")
