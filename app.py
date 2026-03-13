from flask import Flask,request,jsonify
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
le = pickle.load(open("encoder.pkl","rb"))

@app.route("/")
def home():
    return "Weather ML API Running"

@app.route("/predict",methods=["POST"])
def predict():

    data = request.json

    input_data=[[
        data["precipitation"],
        data["temp_max"],
        data["temp_min"],
        data["wind"]
    ]]

    pred=model.predict(input_data)

    result=le.inverse_transform(pred)

    return jsonify({"weather":result[0]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)
