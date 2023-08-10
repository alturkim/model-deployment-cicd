import requests
import json

url = "https://income-prediction-service.onrender.com"

# GET on the root
response = requests.get("https://income-prediction-service.onrender.com")
print(response.status_code)
print(response.json())

# Generate prediction
person = {"age": 43,
          "workclass": "Private",
          "fnlgt": 292175,
          "education": "Masters",
          "education-num": 14,
          "marital-status": "Divorced",
          "occupation": "Exec-managerial",
          "relationship": "Not-in-family",
          "race": "White",
          "sex": "Female",
          "capital-gain": 5,
          "capital-loss": 0,
          "hours-per-week": 45,
          "native-country": "United-States"
        }

response = requests.post("https://income-prediction-service.onrender.com/person", json=person)
print(response.status_code)
print(response.json())
