import requests

API_URL = "http://localhost:3000/api/v1/prediction/0396aa1f-3e1b-4f90-bb6b-fdc1eb46e834"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()
    
output = query({
    "question": "Please analyze the csv file.",
})


print(output)