from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = request.form["gender"]
    education = request.form["education"]
    job = request.form["job"]
    experience = float(request.form["experience"])

    input_data = {
        "Age": age,
        "Experience": experience
    }

    # Initialize all features to 0
    for col in features:
        if col not in input_data:
            input_data[col] = 0

    # Set selected categories to 1
    if f"Gender_{gender}" in features:
        input_data[f"Gender_{gender}"] = 1

    if f"Education_{education}" in features:
        input_data[f"Education_{education}"] = 1

    if f"JobTitle_{job}" in features:
        input_data[f"JobTitle_{job}"] = 1

    final_input = pd.DataFrame([input_data])
    final_input = final_input[features]

    final_input = scaler.transform(final_input)

    prediction = model.predict(final_input)

    return render_template("index.html",
                           prediction_text=f"Predicted Salary: ₹ {round(prediction[0],2)}")

if __name__ == "__main__":
    app.run(debug=True)