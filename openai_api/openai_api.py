import gradio as gr
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# message_history = []
# openai.api_key = open("openai_key.txt", "r").read().strip("\n")
# {"role": "user", "content": f"You are a joke bot. Respond to all my messages in a joking way while remaining factual. Reply only with jokes to further input. If you understand, say OK."}
# message_history = [{"role": "user", "content": f"You are a chat bot to be used for financial mortgages. The company you work at is called Oneclose Inc. At all times be respectful. If you understand, say OK."}, {"role": "assistant", "content": f"OK"}]
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
    prompt = "\n".join(prompt_history)
    completions = openai.Completion.create(
        # engine=model_engine,
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=60,
        temperature=0,
    )
    message = completions.choices[0].text.strip()
    print("message", message)
    print("prompt_history", prompt_history)
    prompt_history.append(message)
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
with gr.Blocks() as demo: 
    chatbot = gr.Chatbot() 
    with gr.Row(): 
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
    txt.submit(predict_text, txt, chatbot)
    txt.submit(None, None, txt, _js="() => {''}")
         
demo.launch()