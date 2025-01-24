from flask import Flask, render_template, request, jsonify
import openai
import datetime
import sys
sys.path.append('../')
from CodeAndData.utils import load_conversation_into_messages
import os
app = Flask(__name__)

# Load the API key
with open("/Users/oscarwan/Research/CodeAndData/chatGPT_API_KEY.txt") as f:
    API_KEY = f.read()

openai.api_key = API_KEY

# Set a default system prompt
default_system_prompt = """You are a recruiter. You evaluate resumes based on the job descriptions given and decide whether to interview the candidate. You give clear decisions and reasons for your decisions."""

# Use a global variable to hold the current system prompt
system_prompt = default_system_prompt

messages = [{"role": "system", "content": system_prompt}]

@app.route('/')
def index():
    # Pass the default system prompt to the template to display to the user
    return render_template('index.html', system_prompt=default_system_prompt)

# Add a new route to update the system prompt
@app.route('/update_prompt', methods=['GET', 'POST'])
def update_prompt():
    global system_prompt  # Use global to modify the system prompt outside of the current scope
    if request.method == 'POST':
        # Get the system prompt from the form
        system_prompt = request.form.get('system_prompt', default_system_prompt)
        messages[0]["content"] = system_prompt  # Update the system prompt in messages
        print("Updated system prompt")
        return jsonify({"status": "success", "message": "System prompt updated"})
    return render_template('update_prompt.html', system_prompt=system_prompt)

current_filename = f"conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')}.txt"

def append_to_conversation_file():
    """Append the latest messages to the current conversation .txt file."""
    filepath = f"./saved_conversations/{current_filename}"  # Assuming you have a directory named 'saved_conversations'
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'a') as file:  # Use 'a' mode to append to the file
        user_message = messages[-2]  # Get the second last message, which is the user's
        bot_message = messages[-1]   # Get the last message, which is the bot's
        
        file.write(f"{user_message['role'].upper()}: {user_message['content']}\n")
        file.write(f"{bot_message['role'].upper()}: {bot_message['content']}\n")
        
@app.route('/load_conversation', methods=['POST'])
def load_conversation():
    conversation_path = request.form['conversation_path']
    
    global messages  # Declare messages as global to modify it
    try:
        messages = load_conversation_into_messages(conversation_path)
        return jsonify({"status": "success", "message": f"Loaded conversation from {conversation_path}", "messages": messages})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_message']
    temperature = float(request.form.get('temperature', 0.7))
    presence_penalty = float(request.form.get('presence_penalty', 0.0))
    selected_model = request.form.get('model', 'gpt-4')  # Default to gpt-4 if not specified

    messages.append({"role": "user", "content": user_input})
    print(f"Generating response using model: {selected_model}") 
    response = openai.ChatCompletion.create(
        model=selected_model,
        messages=messages,
        temperature=temperature,
        presence_penalty=presence_penalty,
        # repetition_penalty=repetition_penalty
    )
    
    bot_response = response.choices[0].message['content']
    messages.append({"role": "assistant", "content": bot_response})
    append_to_conversation_file()
    
    return jsonify({"bot_response": bot_response})


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global messages  # Declare messages as global to modify it
    messages = [{"role": "system", "content": system_prompt}]  # Reset to initial state with system prompt
    return jsonify({"status": "success", "message": "Chat history cleared"})


if __name__ == "__main__":
    app.run(debug=True)
