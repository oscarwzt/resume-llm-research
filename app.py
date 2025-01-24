from flask import Flask, render_template, request, jsonify
import openai
import datetime
from utils import load_conversation_into_messages

app = Flask(__name__)

# Load the API key
with open("/Users/oscarwan/Research/CodeAndData/chatGPT_API_KEY.txt") as f:
    API_KEY = f.read()

openai.api_key = API_KEY

# Set a default system prompt
default_system_prompt = """You are a writing assistant that helps student with writing graduate school application essays.
    You tailor your responses to the student's background and the program they are applying to.
    You ask thoughtful questions to help the student reflect on their experiences and goals."""

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
    # Get the new parameters from the form data
    temperature = float(request.form.get('temperature', 0.7))
    presence_penalty = float(request.form.get('presence_penalty', 0.0))
    repetition_penalty = float(request.form.get('repetition_penalty', 0.0))

    messages.append({"role": "user", "content": user_input})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        presence_penalty=presence_penalty,
        #repetition_penalty=repetition_penalty
    )
    
    bot_response = response.choices[0].message['content']
    messages.append({"role": "assistant", "content": bot_response})
    # Append the latest messages to the current conversation file
    append_to_conversation_file()
    
    return jsonify({"bot_response": bot_response})



if __name__ == "__main__":
    app.run(debug=True)
