from typing import Union

from fastapi import FastAPI, UploadFile, HTTPException, File

import random

import markdown
from pydantic import BaseModel

import os
import openai
import requests
import json
import base64
from weasyprint import HTML

from requests_toolbelt.multipart.encoder import MultipartEncoder

from dotenv import load_dotenv

load_dotenv(".env.local")

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

app = FastAPI()


# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Load the model and tokenizer
# model_name = "biodatlab/score-claim-identification"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# def is_claim(text: str) -> bool:
#     """
#     Determine if the given text is a claim.

#     Args:
#         text (str): The input text.

#     Returns:
#         bool: True if the text is a claim, False otherwise.
#     """
#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    
#     # Get model predictions
#     logits = model(**inputs).logits
    
#     # Get the label with the highest score
#     pred = logits.argmax(dim=1).item()
    
#     # Return True if it's classified as a claim (label 1)
#     return pred == 1

# # Example usage
# text = "We consistently found that participants selectively chose to learn that bad (good) things happened to bad (good) people (Studies 1 to 7) that is, they selectively exposed themselves to deserved outcomes."
# print(f"Is the text a claim? {is_claim(text)}")

GOOGLE_API_URL = f"https://speech.googleapis.com/v1p1beta1/speech:recognize?key={os.environ.get('GOOGLE_API_KEY')}"


def send_to_google_speech_api(audio_file_path):
    # Read the file and encode it in Base64
    with open(audio_file_path, "rb") as f:
        audio_content = base64.b64encode(f.read()).decode("utf-8")

    # Prepare the request payload
    payload = {
        "config": {
            "encoding": "LINEAR16",
            # "sampleRateHertz": 16000,
            "languageCode": "en-US"
        },
        "audio": {
            "content": audio_content
        }
    }

    # Send the request
    response = requests.post(GOOGLE_API_URL, json=payload)

    # Parse the response
    if response.status_code == 200:
        response_data = response.json()
        results = response_data.get("results", [])
        if results:
            return results[0]["alternatives"][0]["transcript"]
        else:
            return "No transcript available."
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Google API error: {response.json()}"
        )


# from pydub import AudioSegment
# sound = AudioSegment.from_wav("recording.wav")
# sound = sound.set_channels(1)
# sound.export("recording2.wav", format="wav")

# print(send_to_google_speech_api("recording2.wav"))

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):


    # Save the uploaded file temporarily
    temp_file_path = f"temp{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    try:
        # Send the file to Google Speech-to-Text API and get the transcript
        transcript = send_to_google_speech_api(temp_file_path)
        
        res = add_transcript(transcript, "me")
        
        return res
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

class Source:
    title: str
    url: str
    publisher: str
    rating: str
    
    def __init__(self, url: str, title: str = None, publisher: str = None, rating: str = None):
        self.title = title
        self.url = url
        self.publisher = publisher
        self.rating = rating
        
    def create_html_link(self):
        return "<a href='" + self.url + "'>" + (self.title if self.title else self.url) + "</a>"

class Message:
    text: str
    user: str
    sources: list
    
    def __init__(self, text: str, user: str, sources = []):
        self.text = text
        self.user = user
        self.sources = sources
        
    def to_json(self):
        return {
            "text": self.text,
            "user": self.user,
            "sources": [source.__dict__ for source in self.sources]
        }

class Transcript:
    id: int
    messages: list
    
    def __init__(self):
        self.id = None
        self.messages = []
        
    def reset(self):
        self.id = random.randint(1000, 9999)
        self.messages = []
    
    def add_message(self, message: Message):
        self.messages.append(message)
        
    def get_transcript(self):
        return self.messages
    
    def to_json(self):
        return {
            "id": self.id,
            "messages": [message.to_json() for message in self.messages]
        }
        
    def pretty(self):
        # create a string in markdown table format user: message\n
        # return markdown.markdown("| User | Message |\n| --- | --- |\n" + "\n".join(["| " + message.user + " | " + message.text.replace("\n", "<br>") + " |" for message in self.messages]), extensions=['tables'], output_format='html5')
        
        # return styled html table with user and message columns
        return "<table><tr><th>User</th><th>Message</th></tr>" + "\n".join(["<tr><td style='padding: 1rem 0; width: 125px; display: flex; flex-direction: column; justify-content: start; align-items: end;'><b>" + message.user + ":</b></td><td style=''>" + markdown.markdown(message.text.replace("\n", "<br>")) + "<br>" + "<br>".join([source.create_html_link() for source in message.sources]) + "</td></tr>" for message in self.messages]) + "</table>"

transcript = Transcript()

@app.post("/start_transcript")
def start_transcript():
    # random id
    transcript.reset()
    
    url = "https://uploads.pinata.cloud/v3/files"
    
    headers = {
        "Authorization": "Bearer " + os.environ.get("PINATA_JWT"),
    }
    
    with open("latest.json", "w") as file:
        json.dump(transcript.to_json(), file)    

    # Use MultipartEncoder to handle the boundary and Content-Type
    multipart_data = MultipartEncoder(
        fields={
            "file": ("latest.json", open("latest.json", "rb"), "application/json"),  # File to upload
        }
    )
    
    os.remove("latest.json")
    
    headers["Content-Type"] = multipart_data.content_type

    response = requests.request("POST", url, data=multipart_data, headers=headers)
    
    print(response.json().get("data").get("cid"))
    
    
    return {"status": "started"}

@app.post("/end_transcript")
def end_transcript():
    # save to pinata
    
    try:
        url = "https://api.pinata.cloud/v3/files"

        querystring = {"name":"latest.json"}

        headers = {"Authorization": "Bearer " + os.environ.get("PINATA_JWT")}

        response = requests.request("GET", url, headers=headers, params=querystring)
    
        for file in response.json().get("data").get("files"):
            
            
            
            print(file.get("cid") + " " + file.get("name"))
            url = "https://api.pinata.cloud/v3/files/" + file.get("id")

            headers = {"Authorization": "Bearer " + os.environ.get("PINATA_JWT")}

            response = requests.request("DELETE", url, headers=headers)
            
            print(response.text)
        
    except:
        
        print("No file to delete")
    
    url = "https://uploads.pinata.cloud/v3/files"
    
    allChats = "\n".join([message.user + ":" + message.text for message in transcript.messages])
    
    title_response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=[
            {"role":"system", "content": "You are a an assistant tasked with generating an 5 word summary from a conversation"},
            {"role":"user", "content": allChats + "\n Using the conversation history summarize the conversation in 5 words or less."}
        ],
        temperature =  0.1,
        top_p = 0.1
    )
    
    
    HTML(string=transcript.pretty()).write_pdf(str(transcript.id) + ".pdf")
    
    headers = {
        "Authorization": "Bearer " + os.environ.get("PINATA_JWT"),
    }
    
    # # Metadata
    metadata = {
        "title": title_response.choices[0].message.content
    }

    # Use MultipartEncoder to handle the boundary and Content-Type
    multipart_data = MultipartEncoder(
        fields={
            "keyvalues": json.dumps(metadata),  # JSON metadata
            "file": (str(transcript.id) + ".pdf", open(str(transcript.id) + ".pdf", "rb"), "application/pdf"),  # File to upload
        }
    )

    # Add Content-Type header with boundary
    headers["Content-Type"] = multipart_data.content_type

    # Make the POST request
    response = requests.post(url, data=multipart_data, headers=headers)
    
    url = "https://uploads.pinata.cloud/v3/files"
    
    headers = {
        "Authorization": "Bearer " + os.environ.get("PINATA_JWT"),
    }
    
    with open(str(transcript.id) + ".json", "w") as file:
        json.dump(transcript.to_json(), file)    

    # # Metadata
    metadata = {
        "title": title_response.choices[0].message.content
    }

    # Use MultipartEncoder to handle the boundary and Content-Type
    multipart_data = MultipartEncoder(
        fields={
            "keyvalues": json.dumps(metadata),  # JSON metadata
            "file": (str(transcript.id) + ".json", open(str(transcript.id) + ".json", "rb"), "application/json"),  # File to upload
        }
    )

    # Add Content-Type header with boundary
    headers["Content-Type"] = multipart_data.content_type

    # Make the POST request
    response = requests.post(url, data=multipart_data, headers=headers)
    
    
    os.remove(str(transcript.id) + ".pdf")
    os.remove(str(transcript.id) + ".json")

    print(response.text)
    
    transcript.reset()
    return {"status": "ended"}

def save_latest_transcript():
    # save to pinata
    try:
        url = "https://api.pinata.cloud/v3/files"

        querystring = {"name":"latest.json"}

        headers = {"Authorization": "Bearer " + os.environ.get("PINATA_JWT")}

        response = requests.request("GET", url, headers=headers, params=querystring)
    
        for file in response.json().get("data").get("files"):
            
            
            
            print(file.get("cid") + " " + file.get("name"))
            url = "https://api.pinata.cloud/v3/files/" + file.get("id")

            headers = {"Authorization": "Bearer " + os.environ.get("PINATA_JWT")}

            response = requests.request("DELETE", url, headers=headers)
            
            print(response.text)
        
    except:
        
        print("No file to delete")
    
    url = "https://uploads.pinata.cloud/v3/files"

    with open("latest.json", "w") as file:
        json.dump(transcript.to_json(), file)    
        # write the transcript to a json file

    # Use MultipartEncoder to handle the boundary and Content-Type
    multipart_data = MultipartEncoder(
        fields={
            # "keyvalues": json.dumps(metadata),  # JSON metadata
            "file": ("latest.json", open("latest.json", "rb"), "application/json"),  # File to upload
        }
    )
    
    os.remove("latest.json")
    
    headers = {"Authorization": "Bearer " + os.environ.get("PINATA_JWT")}
    
    headers["Content-Type"] = multipart_data.content_type

    response = requests.request("POST", url, data=multipart_data, headers=headers)
    
    
    
    
class Transcript(BaseModel):
    message: str
    user: str

@app.post("/transcript")
def add_transcript(message: str, user: str):
    
    transcript.add_message(
        Message(text=message, user=user)
    )
    
    save_latest_transcript()
    
    response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=[
            {"role":"system", "content": "You are a judge trying to determine whether or not a statement is worthy of being fact checked."},
            {"role":"user", "content": message + "\nRegardless of it's validity, is the following statement worthy of being fact checked? Respond with 'yes' or 'no'."}
        ],
        temperature =  0.1,
        top_p = 0.1
    )

    print(response.choices[0].message.content)
    
    if response.choices[0].message.content.lower().startswith("yes"):
    # if is_claim(message):
        
        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "Be precise and concise."
                },
                {
                    "role": "user",
                    "content": "Fact check the following statement: " + message + "\nRespond with two sentences. The first sentence should confirm or deny the claim. The second sentence should provide the truth."
                }
            ],
            "max_tokens": 75,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": "Bearer " + os.environ.get("PERPLEXITY_API_KEY"),
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        result = response.json()
        
        print(json.dumps(result, indent=4))
        
        transcript.add_message(Message(text=result.get("choices")[0].get("message").get("content"), user="Fact Check", sources=[Source(citation) for citation in result.get("citations")]))

        save_latest_transcript()
        
        return {"fact check": result.get("choices")[0].get("message").get("content")}

    return {"fact check": "null"}

@app.get("/latest")
def get_latest_transcript():
    return transcript.to_json()

start_transcript()
add_transcript("There once were thousands of aliens on Earth", "me")
add_transcript("Ants are 10x stronger than humans", "me")
# add_transcript("There once were thousands of aliens on Earth", "me")
end_transcript()