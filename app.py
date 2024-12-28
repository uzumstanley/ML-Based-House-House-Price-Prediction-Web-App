from flask import Flask, request, render_template
import pickle
import numpy as np

# Load your saved model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the Flask app and specify the templates folder
app = Flask(__name__, template_folder='templates2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        bedrooms = float(request.form['bedrooms'])
        sqft_living = float(request.form['sqft_living'])
        condition = float(request.form['condition'])
        yr_built = float(request.form['yr_built'])
        yr_renovated = float(request.form['yr_renovated'])

        # Prepare data for prediction
        features = np.array([[bedrooms, sqft_living, condition, yr_built, yr_renovated]])
        
        # Predict using the loaded model
        prediction = model.predict(features)

        # Render the HTML template with the prediction result
        return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}')
    
    except ValueError as e:
        return render_template('index.html', prediction_text='Error: Please ensure all input values are numeric.')

if __name__ == "__main__":
    app.run(debug=True)


'''
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])
    feature5 = float(request.form['feature5'])

    # Prepare data for prediction
    features = np.array([[feature1, feature2, feature3, feature4, feature5]])
    
    # Predict using the loaded model
    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f'Predicted House Price: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
'''
    
    
'''
# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    try:
        # Extract input features from form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Make prediction using the loaded model
        prediction = model.predict(final_features)
        
        # Output prediction
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
'''
