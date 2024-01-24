from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the machine learning model
model = pickle.load(open('heart.pkl', 'rb'))

# Assuming you have feature names used during training
# Replace this with the actual list of feature names used during training
x = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Define the Flask application
app = Flask(__name__)

# Define API endpoint for prediction


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        json_data = request.json
        query_df = pd.DataFrame(json_data)

        # Ensure feature names consistency
        query_df = pd.get_dummies(query_df)[x]

        # Make prediction
        prediction = model.predict(query_df)

        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Handle exceptions and return an error response if necessary
        return jsonify({'error': str(e)})


# Run the Flask application
if __name__ == '__main__':
    app.run(port=5000, debug=True)
