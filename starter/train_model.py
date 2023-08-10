# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data, clean
from ml.model import train_model, inference, compute_model_metrics
import pandas as pd
import pickle


def calculate_slice_performance(model, df, cat, encoder, lb):
    results = {}
    for value in df[cat].unique():
        df_temp = df[df[cat] == value]
        X_test, y_test, _, _ = process_data(
            df_temp, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        preds = inference(model, X_test)
        results[value] = {k: v for k, v in zip(["precision", "recall", "fbeta"], compute_model_metrics(y_test, preds))}
    return results



if __name__ == "__main__":
    # Add code to load in the data.
    data = pd.read_csv("data/census.csv")
    data = clean(data)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # Train and save a model.
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    print(compute_model_metrics(y_test, preds))
    pickle.dump({"model":model,"encoder":encoder, "lb":lb}, open("model/model.pkl", "wb"))

    # Generate Predictions on Training Data
    train_preds = inference(model, X_train)
    train.reset_index(drop=True, inplace=True)
    bias_analysis_df = pd.concat([train, pd.DataFrame(y_train, columns=["label_value"]),
        pd.DataFrame(train_preds, columns=["score"])], axis=1)
    bias_analysis_df.to_csv("bias_analysis_data.csv")

    # Testing the model on Slice of data
    with open("results/slice_output.txt", "w") as f_out:
        for cat in cat_features:
            f_out.write(f"Performance on Slices of {cat}:\n")
            results = calculate_slice_performance(
                model, test, cat, encoder, lb)
            for cat_value, metrics in results.items():
                f_out.write(f"{cat_value}:\n")
                for m, v in metrics.items():
                    f_out.write(f"{m}:{v:.2f} ")

                f_out.write("\n"+"-"*30+"\n")
            f_out.write("\n")
