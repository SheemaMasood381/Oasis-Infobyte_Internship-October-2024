from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('Random_Forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request (assumed to be in JSON format)
        data = request.get_json()

        # Check if all required fields are present
        required_fields = ['TV', 'Radio', 'Newspaper']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing data fields.'}), 400

        # Extract features from the incoming JSON
        TV = data['TV']
        Radio = data['Radio']
        Newspaper = data['Newspaper']

        # Create DataFrame for the features
        features = pd.DataFrame({
            'TV': [TV],
            'Radio': [Radio],
            'Newspaper': [Newspaper]
        })

        # Ensure DataFrame columns match model's expected columns
        features = features[['TV', 'Radio', 'Newspaper']]

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict sales
        prediction = model.predict(scaled_features)[0]

        # Return prediction as JSON
        return jsonify({'predicted_sales': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
