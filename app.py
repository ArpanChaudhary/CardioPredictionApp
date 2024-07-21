import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('LR_CardioML.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Check the types of the loaded objects
print(f"Model type: {type(model)}")
print(f"Scaler type: {type(scaler)}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    data_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(data_array)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text="Cardio Disease: {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
