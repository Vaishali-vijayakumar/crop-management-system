import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv('top_20_commodities.csv')

# Encode categorical variables
le_market = LabelEncoder()
le_commodity = LabelEncoder()
data['Market'] = le_market.fit_transform(data['Market'])
data['Commodity'] = le_commodity.fit_transform(data['Commodity'])

# Define features and target
X = data[['Market', 'Commodity']]
y = data['Modal_Price']

# Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the model and encoders
with open('price_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'Market': le_market, 'Commodity': le_commodity}, f)

print("Model and encoders saved successfully!")
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import json

app = Flask(__name__)

# Load the trained model and label encoders
with open('price_predictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load the dataset to display commodity data
commodity_data = pd.read_csv('top_20_commodities.csv')

@app.route('/')
def home():
    # Extract unique markets and commodities for dropdowns
    commodities = commodity_data['Commodity'].unique().tolist()
    markets = commodity_data['Market'].unique().tolist()
    return render_template('price.html', commodities=commodities, markets=markets)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data
    data = request.json
    commodity = data.get('commodity')
    market = data.get('market')

    if not commodity or not market:
        return jsonify({'error': 'Both commodity and market must be provided'}), 400

    try:
        # Encode inputs using the label encoders
        commodity_encoded = label_encoders['Commodity'].transform([commodity])[0]
        market_encoded = label_encoders['Market'].transform([market])[0]

        # Make the prediction
        input_data = pd.DataFrame([[market_encoded, commodity_encoded]], columns=['Market', 'Commodity'])
        predicted_price = model.predict(input_data)[0]

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

