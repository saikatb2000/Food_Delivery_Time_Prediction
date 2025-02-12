from flask import Flask, request, render_template
import joblib
import pandas as pd
from ml_helper import apply_transformer, create_speed

# Load the trained model and data
model = joblib.load("regression_model.joblib")
X_train = pd.read_csv('X_train', index_col=0)

# Initialize the Flask app
app = Flask(__name__)


@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction logic when form is submitted."""
    try:
        # Get user inputs from the form
        distance = float(request.form.get('distance', 0))
        weather = request.form.get('weather', "")
        traffic = request.form.get('traffic', "")
        time_of_day = request.form.get('timeofday', "")
        vehicle = request.form.get('VT', "")
        prep_time = int(request.form.get('pre_time', 0))

        # Validate user inputs
        if not weather or not traffic or not time_of_day or not vehicle:
            raise ValueError("All fields are required.")

        # Create the input DataFrame
        input_data = pd.DataFrame({
            "Distance_km": [distance],
            "Preparation_Time_min": [prep_time],
            "Weather": [weather],
            "Traffic_Level": [traffic],
            "Time_of_Day": [time_of_day],
            "Vehicle_Type": [vehicle]
        })

        # Apply custom transformations
        input_data = create_speed(input_data)
        transformed_data = apply_transformer(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)
        predicted_time = prediction.item()  # Extract scalar value from NumPy array
        rounded_time = round(predicted_time, 2)  # Round the value

        return render_template(
            'index.html',
            prediction_text=f'Estimated Delivery Time: {rounded_time} minutes'
        )
    except ValueError as ve:
        # Handle invalid inputs
        return render_template('index.html', prediction_text=f'Error: {ve}')
    except Exception as e:
        # Handle other unexpected errors
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
