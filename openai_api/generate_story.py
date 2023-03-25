import boto3
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
import subprocess
import re
import time
import textwrap
# voice_id = "Ruth"
voice_id = "Matthew"
output_format = "mp3"

# Create an Amazon Polly client
polly_client = boto3.Session(
    aws_access_key_id='AKIAWN6557C6L2SLN77M',
    aws_secret_access_key='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',
    region_name='ca-central-1').client('polly')

message_history = [{"role": "user", "content": """You are an interactive story game bot that proposes some hypothetical fantastical situation where the user needs to pick from 2-4 options that you provide. Once the user picks one of those options, you will then state what happens next and present new options, and this then repeats. If you understand, say, OK, and begin when I say "begin." When you present the story and options, present just the story and start immediately with the story, no further commentary, and then options like "Option 1:" "Option 2:" ...etc. End the story after 3 rounds of options. Don't start the story with Great or anything like that."""},
                                      {"role": "assistant", "content": f"""OK, I understand. Begin when you're ready."""}]
message_history.append({"role": "user", "content": "OK, begin."})

def generate_story(input):
  start_time = time.time()
  completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=input)
  reply_content = completion.choices[0].message.content
  print(f"Generate Story Took: {time.time() - start_time}")
  print(reply_content)
  return reply_content


def generate_audio(input):
  start_time = time.time()
  audio = polly_client.synthesize_speech(
      Text=input,
      VoiceId=voice_id,
      OutputFormat=output_format)
  print(f"Generate Audio Took: {time.time() - start_time}")
  return audio

def save_audio(response, count):
  with open(f"output{count}.mp3", "wb") as f:
      f.write(response["AudioStream"].read())

def read_audio(file):
  command = ["C:\\Program Files (x86)\\Windows Media Player\\wmplayer.exe", '/play', '/close', f"C:/Users/Marinus/OneDrive/Desktop/Development/Playground/openai_api/{file}"]
  process = subprocess.Popen(command)
  return process

count = 0
for step in range(5):
  story = generate_story(message_history)
  audio = generate_audio(story)
  save_audio(audio, count)
  media_player = read_audio(f"output{count}.mp3")
  while True:
    user_input = input("You: ")
    if re.search(r"\S", user_input):
      media_player.terminate()
      lines = story.split('\n')
      # filtered_lines = [line for line in lines if line.startswith(f"Option {user_input}:")]
      filtered_lines = [line for line in lines if not (line.startswith("Option") and not line.startswith(f"Option {user_input}:"))]
      new_text = '\n'.join(filtered_lines)
      message_history.append({"role": "assistant", "content": new_text})
      message_history.append({"role": "user", "content": user_input})
      count += 1
      print("\n")
      break
    else:
      print("Invalid input, please try again")
print("End of story")
for i, message in enumerate(message_history):
  # print(f"{message['role'].capitalize()} {i+1}:")
  if message['role'] == "assistant":
    wrapped_content = textwrap.fill(message['content'], width=60)
    print(wrapped_content)
    print()




