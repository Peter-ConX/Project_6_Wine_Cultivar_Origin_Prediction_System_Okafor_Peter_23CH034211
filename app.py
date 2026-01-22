from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model/wine_cultivar_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        alcohol = float(request.form["alcohol"])
        malic_acid = float(request.form["malic_acid"])
        ash = float(request.form["ash"])
        magnesium = float(request.form["magnesium"])
        color_intensity = float(request.form["color_intensity"])
        hue = float(request.form["hue"])

        data = np.array([[alcohol, malic_acid, ash,
                          magnesium, color_intensity, hue]])
        data = scaler.transform(data)

        result = model.predict(data)[0]
        prediction = f"Cultivar {result + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
