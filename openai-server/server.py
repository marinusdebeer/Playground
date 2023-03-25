from flask import Flask, jsonify, request
import openai
import os
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

message_history = [{"role": "system", "content": f"You are a chat bot to be used for financial mortgages. The company you work at is called Oneclose Inc. At all times be respectful."}]

prompt_history = []

def predict(input):
    message_history.append({"role": "user", "content": f"{input}"})
    # model="gpt-3.5-turbo",#10x cheaper than davinci, and better. $0.002 per 1k tokens
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=message_history
    )
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"}) 
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(1, len(message_history)-1, 2)]  # convert to tuples of list
    return response


def predict_text(input):
    global prompt_history
    prompt_history.append(input)
    print(prompt_history)
    completions = openai.Completion.create(
        # engine="davinci-codex",
        model="text-davinci-003",
        prompt="\n".join(prompt_history),
        max_tokens=60,
        temperature=0.7,
        # stop="\n"
    )
    message = completions.choices[0].text.strip()
    
    prompt_history.append(message)
    print("message", message)
    print("prompt_history", prompt_history)
    response = [(prompt_history[i], prompt_history[i+1]) for i in range(0, len(prompt_history)-1, 2)]
    return response

def create_image(input):
    #returns urls of the images the just open it in a new tab or display somehow
    image = openai.Image.create(
        prompt=input,
        n=2,
        size="1024x1024"
        )
    print(image.data)
    
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400

    prompt = data['prompt']

    try:
        message = predict_text(prompt)
        return jsonify({'message': message}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error processing the request'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT', 3000))



