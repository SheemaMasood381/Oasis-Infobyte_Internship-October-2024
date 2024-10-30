from flask import Flask, render_template, request
import datetime
import pandas as pd
import pickle

app = Flask(__name__)
# Load the trained RandomForestRegressor model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the ColumnTransformer
with open('column_transformer.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Load the SequentialFeatureSelector
with open('sfs.pkl', 'rb') as sfs_file:
    sfs = pickle.load(sfs_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    if request.method == 'POST':
        try:
            present_price = float(request.form['present_price'])
            driven_kms = float(request.form['driven_kms'])
            year = int(request.form['year'])
            fuel_type = request.form['fuel_type']
            selling_type = request.form['selling_type']
            transmission = request.form['transmission']
            owner = int(request.form['owner'])
            
            # Calculate the car's age
            current_year = datetime.datetime.now().year
            car_age = current_year - year

            # Predict car price
            predicted_price = predict_car_price(present_price, driven_kms, car_age, fuel_type, selling_type, transmission, owner)
        except Exception as e:
            print(f"Error: {e}")
    
    return render_template('index.html', predicted_price=predicted_price)

def predict_car_price(present_price, driven_kms, car_age, fuel_type, selling_type, transmission, owner):
    data = {
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Car_Age': [car_age],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner]
    }

    input_df = pd.DataFrame(data)
    transformed_data = preprocessor.transform(input_df)
    selected_features = sfs.transform(transformed_data)
    estimated_price = model.predict(selected_features)[0]

    return estimated_price

if __name__ == '__main__':
    app.run(debug=True)
