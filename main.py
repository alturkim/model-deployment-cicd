# Put the code for your API here.
from fastapi import FastAPI

from pydantic import BaseModel, Field, ConfigDict
import pickle
import pandas as pd

from starter.ml.model import inference
from starter.ml.data import process_data
from fastapi.encoders import jsonable_encoder


class Person(BaseModel):
    class Config:
        allow_population_by_field_name = True
    age: int = Field(gt=0)
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 43,
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
            ]
        }
    }


app = FastAPI()


@app.get("/")
async def welcome_message():
    return {"Greetings": "Welcome to Income Prediction Model API"}


@app.post("/person/")
async def predict(person: Person):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    artifact_dict = pickle.load(open("model/model.pkl", "rb"))
    model = artifact_dict["model"]
    encoder = artifact_dict["encoder"]
    lb = artifact_dict["lb"]

    df = pd.DataFrame.from_dict([Person.parse_obj(person).dict(by_alias=True)])
    df = df.drop(columns="model_config")
    X_test, _, _, _ = process_data(
        df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_test)
    return {"prediction": preds.item()}
