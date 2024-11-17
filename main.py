from typing import Union

from fastapi import FastAPI

import random

import markdown

import os
import openai
import requests
import json
from weasyprint import HTML

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
            "messages": [message.__dict__ for message in self.messages]
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
    return {"status": "started"}

@app.post("/end_transcript")
def end_transcript():
    # save to pinata
    
    # url = "https://api.pinata.cloud/data/testAuthentication"

    # headers = {"Authorization": "Bearer " + os.environ.get("PINATA_JWT")}

    # response = requests.request("GET", url, headers=headers)

    # print(response.text)
    
    url = "https://uploads.pinata.cloud/v3/files"

    # payload = "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n<string>\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"group_id\"\r\n\r\n<string>\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"keyvalues\"\r\n\r\n{}\r\n-----011000010111000001101001--\r\n\r\n"
    
    # save the transcript as a pdf file and upload it to pinata
    
    print(transcript.to_json())
    
    HTML(string=transcript.pretty()).write_pdf(str(transcript.id) + ".pdf")
    
    payload = {
        "file": open( str(transcript.id) + ".pdf", "rb")
    }
    
    # delete the pdf file
    # os.remove(str(transcript.id) + ".pdf")
    
    headers = {
        "Authorization": "Bearer " + os.environ.get("PINATA_JWT"),
        # "Content-Type": "multipart/form-data"
    }

    response = requests.request("POST", url, data = {'key': 'value'}, files=payload, headers=headers)

    print(response.text)
    
    transcript.reset()
    return {"status": "ended"}

@app.post("/transcript")
def add_transcript(message: str, user: str):
    transcript.add_message(
        Message(text=message, user=user)
    )
    
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
        
        # fact check the message using google fact check tools
        # res = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search?query=" + message + "&key=" + os.environ.get("GOOGLE_API_KEY"))
        
        # print(os.environ.get("GOOGLE_API_KEY"))
        # json = res.json()
        
        # claims = json.get("claims")
        
        # if not claims:
        #     print("No relevant sources found.")
        #     return {"fact check": "No relevant sources found."}
        
        # # print(claims[0])
        # # print(claims[0].get("text"), claims[0].get("claimReview")[0].get("textualRating"))
        
        # claim_context = "\n".join(["Claim: " + claims[0].get("text"), "Rating: " + claims[0].get("claimReview")[0].get("textualRating")])
        
        # response = client.chat.completions.create(
        #     model='Meta-Llama-3.1-8B-Instruct',
        #     messages=[
        #         {"role":"system", "content": "You are an objective observer and your job is to condense the information into a brief summary."},
        #         {"role":"user", "content": claim_context + "\nCompile the results of the previous claims and their ratings to serve as knowledge. Respond only to the claim in the following message in 2 sentences or less:\n" + message} 
        #     ],
        #     temperature =  0.1,
        #     top_p = 0.1
        # )

        # print(response.choices[0].message.content)
        
        # transcript.add_message(Message(text=response.choices[0].message.content, user="Fact Check", sources=[Source(claim.get("claimReview")[0].get("title"), claim.get("claimReview")[0].get("url"), claim.get("claimReview")[0].get("publisher").get("name"), claim.get("claimReview")[0].get("textualRating")) for claim in claims]))
        
        # return {"fact check": response.choices[0].message.content}

    return {"fact check": None}

start_transcript()
add_transcript("There once were thousands of dragons on Earth", "Speaker 1")
end_transcript()