
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

# with open('best_model.pkl', 'rb') as file:
#     clf = pickle.load(file)

# prediction = clf.predict([[0,5.1, 3.5, 1.4, 0.2]])
# prediction

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = request.form
        gender = int(data['gender']) 
        hemoglobin = float(data['hemoglobin'])
        mch = float(data['mch'])
        mchc = float(data['mchc'])
        mcv = float(data['mcv'])

        features = np.array([[gender, hemoglobin, mch, mchc, mcv]])

        prediction = model.predict(features)

        output = prediction
        return render_template('index.html', prediction_text=format(output[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
# flask part is done



