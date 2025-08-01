from flask import Flask, request, render_template
import pandas as pd
import joblib
from data_processing import preprocess_data

app = Flask(__name__)

# Load your trained model
model = joblib.load("model/heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        form_data = request.form.to_dict()

        # Convert appropriate values
        for key in form_data:
            try:
                form_data[key] = float(form_data[key])
                if form_data[key].is_integer():
                    form_data[key] = int(form_data[key])
            except:
                pass  # leave string values as-is

        # Create DataFrame
        input_df = pd.DataFrame([form_data])

        # Preprocess input
        processed = preprocess_data(input_df, is_train=False)

        # Predict
        prediction = model.predict(processed)[0]
        result = "Likely to have Heart Disease" if prediction == 1 else "No Heart Disease Detected"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
