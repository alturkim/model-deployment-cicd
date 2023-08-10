import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, clean
from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv("data/census.csv")
    df = clean(df)
    train, test = train_test_split(df, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="session")
def model(data):
    X_train, y_train, _, _ = data
    model = train_model(X_train, y_train)
    return model


def test_train_model(data, model):
    X_train, y_train, _, _ = data
    # model = train_model(X_train, y_train)
    assert model is not None


def test_compute_model_metrics(data, model):
    X_train, y_train, X_test, y_test = data
    # model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(data, model):
    X_train, _, _, _ = data
    preds = inference(model, X_train)
    assert preds is not None
