import pickle

from flask import Flask, render_template, request

app = Flask(__name__)

# load the model from disk
filename = 'marriage_age_predict_model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    gender = int(request.args['gender'])
    religion = int(request.args['religion'])
    caste = int(request.args['caste'])
    mother_tongue = int(request.args['mother_tongue'])
    country = int(request.args['country'])
    height_cms = int(request.args['height_cms'])

    predicted_age = model.predict([gender, religion, caste, mother_tongue, country, height_cms, ])

    print("checking   ", str(round(predicted_age[0], 2)))
    return str(round(predicted_age[0], 2))


if __name__ == "__main__":
    app.run(port=2000, debug=True)
