from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import gzip

app = Flask(__name__)

# Set the template folder explicitly
current_folder = os.path.abspath(os.getcwd())  # Get the current working directory
template_folder_path = os.path.join(current_folder, 'C:/Users/Shrinath/AIML_Shrinath_practice/Asktalos_Project/VS_code/templates')
app = Flask(__name__, template_folder=template_folder_path)

# Path to your gzipped model file
gzipped_model_path = 'C:/Users/Shrinath/AIML_Shrinath_practice/Asktalos_Project/VS_code/compressed_model.pkl.gz'

# Function to load the compressed pickle file and return the model
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

# Load the compressed model
loaded_model = load_compressed_pickle(gzipped_model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        bootstrapped_data = {
            'Formulation Duration (hrs)': float(request.form['Formulation_Duration']),
            'Additive Catalyst (gm)': float(request.form['Additive_Catalyst']),
            'Plasticizer (gm)': float(request.form['Plasticizer']),
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([bootstrapped_data])

        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_df)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)