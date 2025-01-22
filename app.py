from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

# Load MinMaxScaler (you should use the same scaler used in training)
scaler = MinMaxScaler()
# scaler = joblib.load('scaler.pkl')  # If you saved it during training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form data and convert to float
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pm25 = float(request.form['pm25'])
            pm10 = float(request.form['pm10'])
            co = float(request.form['co'])
            no2 = float(request.form['no2'])
            so2 = float(request.form['so2'])  # New input for SO2
            pia = float(request.form['pia'])  # New input for Proximity to Industrial Areas
            pd = float(request.form['pd'])  # New input for Population Density

            # Create input data as a list of values
            input_data = [
                [temperature, humidity, pm25, pm10, co, no2, so2, pia, pd]
            ] 

            # Create DataFrame using the correct format
            input_df = pd.DataFrame(
                input_data, 
                columns=[
                    'Temperature', 'Humidity', 'PM2.5', 'PM10', 'CO', 'NO2', 
                    'SO2', 'Proximity_to_Industrial_Areas', 'Population_Density'
                ]
            )

            # Scale the data (use the same scaler as in training)
            input_scaled = scaler.transform(input_df)

            # Get model prediction
            prediction = model.predict(input_scaled)

            # Convert prediction to a more readable form (optional)
            air_quality = 'Good' if prediction[0] == 0 else 'Poor'  # Assuming 0=Good, 1=Poor

            return render_template('index.html', prediction=air_quality)

        except Exception as e:
            return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)