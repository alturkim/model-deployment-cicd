from fastapi.testclient import TestClient

from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

def test_api_welcome_message_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Greetings": "Welcome to Income Prediction Model API"}


def test_positive_prediction():
    response = client.post(
        "/person/",
        json={"age": 43,
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
                    "native-country": "United-States"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1
    }

def test_negative_prediction():
    response = client.post(
        "/person/",
        json={"age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": " Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0
    }
