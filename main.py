from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load models and encoders for yield prediction
with open('yield_predictor_model.pkl', 'rb') as f:
    yield_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    yield_encoders = pickle.load(f)

# Load models and encoders for price prediction
with open('price_predictor_model.pkl', 'rb') as f:
    price_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    price_encoders = pickle.load(f)

# Load commodity dataset for price prediction
commodity_data = pd.read_csv('top_20_commodities.csv')

@app.route('/')
def home():
    """Home page displaying links to Yield Prediction and Price Prediction."""
    return render_template('home.html')

@app.route('/yield')
def yield_home():
    """Render yield prediction page."""
    return render_template('yield.html', crops=yield_encoders['Crop'].classes_)

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        # Parse input data from the JSON request
        data = request.json
        acre = float(data.get('acre', 0))
        crop = data.get('crop')
        rainfall = float(data.get('rainfall', 0))
        humidity = float(data.get('humidity', 0))
        temperature = float(data.get('temperature', 0))
        soil_ph = float(data.get('soilPh', 0))
        fertilizer = float(data.get('fertilizer', 0))

        # Encode categorical data (crop)
        if crop not in yield_encoders['Crop'].classes_:
            raise ValueError("Invalid crop value")
        crop_encoded = yield_encoders['Crop'].transform([crop])[0]

        # Prepare input features for prediction
        features = [[acre, crop_encoded, rainfall, humidity, temperature, soil_ph, fertilizer]]

        # Predict yield using the model
        predicted_yield = yield_model.predict(features)[0]

        # Return prediction results
        return jsonify({
            "crop": crop,
            "acre": acre,
            "predicted_yield": round(predicted_yield, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/price')
def price_home():
    """Render price prediction page."""
    commodities = commodity_data['Commodity'].unique().tolist()
    markets = commodity_data['Market'].unique().tolist()
    return render_template('price.html', commodities=commodities, markets=markets)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Parse the input data
        data = request.json
        commodity = data.get('commodity')
        market = data.get('market')

        if not commodity or not market:
            return jsonify({'error': 'Both commodity and market must be provided'}), 400

        # Encode inputs using the label encoders
        commodity_encoded = price_encoders['Commodity'].transform([commodity])[0]
        market_encoded = price_encoders['Market'].transform([market])[0]

        # Make the prediction
        input_data = pd.DataFrame([[market_encoded, commodity_encoded]], columns=['Market', 'Commodity'])
        predicted_price = price_model.predict(input_data)[0]

        # Filter the commodity data for the selected market and commodity
        filtered_data = commodity_data[
            (commodity_data['Market'] == market) & (commodity_data['Commodity'] == commodity)
        ].to_dict(orient='records')

        return jsonify({
            'commodity': commodity,
            'market': market,
            'predicted_price': round(predicted_price, 2),
            'filtered_data': filtered_data
        })

    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
