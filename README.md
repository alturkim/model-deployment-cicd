# Deploying an ML Model to Cloud with CI/CD and FastAPI

This project builds an income classifier based on the publicly available [Census Bureau Dataset](https://archive.ics.uci.edu/ml/datasets/census+income). The project incorporates automation through CI/CD. Furthermore, the produced model is checked for bias using **Slice Testing**. Finally the model is deployed to cloud using **FastAPI** on [Render](https://render.com) hosting service.

## CI/CD
The project is setup with **Continues Integration (CI)** using **Github Actions** to run **pytest** for **unit testing** and **API testing**, and **flake8** for **linting**. Furthermore, **Continues Deployment (CD)** is configured on the cloud to automatically deploy the application after every push to the repository provided the CI passes.

## Model Card
See the [Model Card](model_card.md) for details about the model training process, performance, biases, and caveats.