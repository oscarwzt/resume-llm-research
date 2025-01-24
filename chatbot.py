import openai

# Initialize the API with your key
with open("chatGPT_API_KEY.txt") as f:
    API_KEY = f.read()
    
openai.api_key = API_KEY

def chat_with_gpt4(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    return response.choices[0].message['content']

if __name__ == "__main__":
    print("GPT-4 Chatbot initialized!")
    
    system_prompt = """You are a writing assistant that helps student with writing graduate school application essays.
    You tailor your responses to the student's background and the program they are applying to."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        user_input = input("You: ")
        
        # Append user input to messages
        messages.append({"role": "user", "content": user_input})
        
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        
        response = chat_with_gpt4(messages)
        messages.append({"role": "assistant", "content": response})
        
        print("Chatbot:", response)
