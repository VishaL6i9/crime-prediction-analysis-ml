from flask import Flask, render_template, request, jsonify
import joblib
from datetime import datetime

app = Flask(__name__)

# Load encoders and model
try:
    encoders = joblib.load('encoders.pkl')  # Dict of LabelEncoders
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading model files: {e}")
    encoders = None
    model = None

@app.route('/')
def home():
    # Extract categories using .classes_ from LabelEncoder
    description_categories = encoders['Description'].classes_.tolist() if encoders else []
    primary_type_categories = encoders['Primary Type'].classes_.tolist() if encoders else []
    location_categories = encoders['Location Description'].classes_.tolist() if encoders else []

    return render_template('index.html',
                           description_categories=description_categories,
                           primary_type_categories=primary_type_categories,
                           location_categories=location_categories)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not encoders:
        return render_template('index.html',
                               prediction_text="Error: Model not loaded properly")

    try:
        # Collect form inputs
        form_data = request.form

        # Parse inputs
        description = form_data['description']
        primary_type = form_data['primary_type']
        location_description = form_data['location_description']
        block = form_data['block']
        domestic = 1 if form_data['domestic'].lower() in ["true", "yes", "1"] else 0
        ward = float(form_data['ward'])

        date_obj = datetime.strptime(form_data['date'], '%Y-%m-%d')
        date = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"

        district = float(form_data['district'])
        beat = float(form_data['beat'])
        time = form_data['time']

        # Encode categorical inputs
        encoded_inputs = [
            encoders['Date'].transform([date])[0],
            encoders['Block'].transform([block])[0],
            encoders['Primary Type'].transform([primary_type])[0],
            encoders['Description'].transform([description])[0],
            encoders['Location Description'].transform([location_description])[0],
            domestic,
            beat,
            district,
            ward,
            encoders['Time'].transform([time])[0]
        ]

        # Make prediction
        prediction = model.predict([encoded_inputs])[0]
        probability = round(model.predict_proba([encoded_inputs])[0][1], 2)

        prediction_text = f"Prediction: {prediction}<br>Probability: {probability}"

    except ValueError as ve:
        prediction_text = f"Invalid input format: {ve}"
    except Exception as e:
        prediction_text = f"Error during prediction: {str(e)}"

    return render_template('index.html',
                           prediction_text=prediction_text,
                           description_categories=encoders['Description'].classes_.tolist(),
                           primary_type_categories=encoders['Primary Type'].classes_.tolist(),
                           location_categories=encoders['Location Description'].classes_.tolist())

if __name__ == '__main__':
    app.run(debug=True)
