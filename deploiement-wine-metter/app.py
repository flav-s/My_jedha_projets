from flask import Flask,request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
#https://wine-o-flavs.herokuapp.com/
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)
redis = Redis(host='redis-master', port=5000)
cols =['fixed_acidity','volatile_acidity' , 'citric_acid' , 'total_sulfur_dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol' , 'residual_sugar' , 'chlorides' , 'free_sulfur_dioxide']


@app.route('/',methods=['GET'])
def home():
    #try:
    #    count = redis.incr('hits')
    #except RedisError:
    #    count = ""
    #    print(count)
    return render_template("home1.html")#, cnt ="Bonjour vous êtes le {}ème visiteur !")#.format(count))


@app.route("/home2", methods=["GET", "POST"])
def docu():
    if request.method == "GET":
        return render_template("home2.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_f = pd.DataFrame([final])#, columns = cols)
    prediction = model.predict(data_f)
    prediction = int(prediction[0])
    print('c est la prevision : ', prediction)
    return render_template('home1.html', pred='Ce vin est de qualité : {} / 10'.format(prediction))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(data["input"])
    output = [float(x) for x in prediction]
    return jsonify({"predictions": output}),200


if __name__ == '__main__':
    app.run(debug=True)