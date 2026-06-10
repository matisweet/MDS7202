import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI

with open("models/best_model.pkl", "rb") as file:
    model = pickle.load(file)


app = FastAPI()


def make_prediction(data: dict):
    features = pd.DataFrame(
        [
            {
                "ph": data["pH"],
                "Hardness": data["Hardness"],
                "Solids": data["Solids"],
                "Chloramines": data["Chloramines"],
                "Sulfate": data["Sulfate"],
                "Conductivity": data["Conductivity"],
                "Organic_carbon": data["Organic_carbon"],
                "Trihalomethanes": data["Trihalomethanes"],
                "Turbidity": data["Turbidity"],
            }
        ]
    )

    prediction = model.predict(features).item()

    return int(prediction)


@app.get("/")
async def home():
    return {
        "message": "API para predecir potabilidad del agua",
        "problem": "Clasificar si una muestra de agua es potable o no potable.",
        "input": [
            "pH",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ],
        "output": "potabilidad: 1 si es potable, 0 si no es potable",
    }


@app.post("/potabilidad/")
async def potabilidad(data: dict):
    prediction = make_prediction(data)

    return {"potabilidad": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000)
