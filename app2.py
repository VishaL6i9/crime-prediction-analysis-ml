import tkinter as tk
from tkinter import ttk
import joblib
from datetime import datetime

# Load encoders and model
encoders = joblib.load('encoders.pkl')
model = joblib.load('model.pkl')
#y_pred_prob = model.predict_proba(X_test)[:, 1]
def predict():
    try:
        # Collect inputs
        description = description_var.get()
        primary_type = primary_type_var.get()
        location_description = location_description_var.get()
        block = block_var.get()
        domestic = 1 if domestic_var.get().lower() == "true" else 0
        ward = float(ward_var.get())
        date = date_var.get()
        district = float(district_var.get())
        beat = float(beat_var.get())
        time = time_var.get()

        # Handle date and time conversion

        try:
            time_parsed = datetime.strptime(time, "%H:%M")
        except ValueError as e:
            result_label.config(text=f"Invalid time format. Please use HH:MM. Error: {e}")
            return

        # Encode categorical inputs using the loaded encoders
        date_encoded = encoders['Date'].transform([date])[0]
        description_encoded = encoders['Description'].transform([description])[0]#if description in encoders['Description'].classes_ else -1
        primary_type_encoded = encoders['Primary Type'].transform([primary_type])[0]# if primary_type in encoders['Primary Type'].classes_ else -1
        location_description_encoded = encoders['Location Description'].transform([location_description])[0]# if location_description in encoders['Location Description'].classes_ else -1
        block_encoded = encoders['Block'].transform([block])[0]# if block in encoders['Block'].classes_ else -1
        time_encoded = encoders['Time'].transform([time])[0]
        # Prepare input for the model
        encoded_inputs = [
            date_encoded,
            block_encoded,
            primary_type_encoded,
            description_encoded,
            location_description_encoded,
            domestic,
            beat,
            district,
            ward,
            time_encoded
        ]
        prediction = model.predict([encoded_inputs])[0]
        probabilities = model.predict_proba([encoded_inputs])[0][1] 
        probabilities = round(probabilities, 2)  


        result_label.config(text=f'Prediction: {prediction}\nProbability: {probabilities}')



    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Create GUI window
root = tk.Tk()
root.title("Crime Prediction System")

# Input fields
description_var = tk.StringVar()
primary_type_var = tk.StringVar()
location_description_var = tk.StringVar()
block_var = tk.StringVar()
domestic_var = tk.StringVar()
ward_var = tk.StringVar()
date_var = tk.StringVar()
district_var = tk.StringVar()
beat_var = tk.StringVar()
time_var = tk.StringVar()

fields = [
    ('Description', description_var),
    ('Primary Type', primary_type_var),
    ('Location Description', location_description_var),
    ('Block', block_var),
    ('Domestic (Yes/No)', domestic_var),
    ('Ward', ward_var),
    ('Date', date_var),
    ('District', district_var),
    ('Beat', beat_var),
    ('Time (HH:MM)', time_var)
]

for i, (label, var) in enumerate(fields):
    ttk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='w')
    ttk.Entry(root, textvariable=var).grid(row=i, column=1, padx=10, pady=5)

# Predict button
ttk.Button(root, text="Predict", command=predict).grid(row=len(fields), column=0, columnspan=2, pady=10)

# Result label
result_label = ttk.Label(root, text="Prediction will appear here.")
result_label.grid(row=len(fields) + 1, column=0, columnspan=2)

root.mainloop()

