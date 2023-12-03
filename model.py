from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os
import gzip

app = Flask(__name__)

# Set the template folder explicitly
current_folder = os.path.abspath(os.getcwd())  # Get the current working directory
template_folder_path = os.path.join(current_folder, 'C:/Users/Shrinath/AIML_Shrinath_practice/Asktalos_Project/Asktalos_Final_vscode/templates')
app = Flask(__name__, template_folder=template_folder_path)

# Path to your gzipped model file
gzipped_model_path = 'C:/Users/Shrinath/AIML_Shrinath_practice/Asktalos_Project/Asktalos_Final_vscode/random_forest_model.pkl.gz'

# Function to load the compressed pickle file and return the model
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

# Load the compressed model
loaded_model = load_compressed_pickle(gzipped_model_path)

# Load the scaler for input features
input_scaler_path = 'C:/Users/Shrinath/AIML_Shrinath_practice/Asktalos_Project/Asktalos_Final_vscode/features_scaler.pkl'
with open(input_scaler_path, 'rb') as input_scaler_file:
    input_scaler = pickle.load(input_scaler_file)

# Load the scaler for the target variable
target_scaler_path = 'C:/Users/Shrinath/AIML_Shrinath_practice/Asktalos_Project/Asktalos_Final_vscode/target_scaler.pkl'
with open(target_scaler_path, 'rb') as target_scaler_file:
    target_scaler = pickle.load(target_scaler_file)

# Define the columns to scale
columns_to_scale = ['Formulation Duration (hrs)', 'Additive Catalyst (gm)', 'Plasticizer (gm)']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_data = {
            'Formulation Duration (hrs)': float(request.form['Formulation_Duration']),
            'Additive Catalyst (gm)': float(request.form['Additive_Catalyst']),
            'Plasticizer (gm)': float(request.form['Plasticizer']),
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the input data using the loaded scaler
        scaled_input = input_scaler.transform(input_df[columns_to_scale])

        # Make predictions using the loaded model
        scaled_prediction = loaded_model.predict(scaled_input)[0]

        # Inverse transform the predicted value for the target variable
        prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        return render_template('index.html', input_data=input_data, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)