import requests

API_URL = "http://localhost:3000/api/v1/prediction/85ca04f1-94c1-4366-8183-a270e07a61dc"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()
    
output = query({
    "question": "Hey, how are you?",
})

print(output)