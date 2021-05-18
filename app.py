from flask import Flask, render_template
from flask import request
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')


@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route("/kidney")
def kidney():
    return render_template("kidney.html")


@app.route('/disbetes_result', methods=["POST"])
def disbetes_result():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(float, predict_list))
        print(predict_list)
        predict = np.array(predict_list).reshape(1, 5)
        loaded_model = pickle.load(open("diabetes_model.pkl", 'rb'))
        result = loaded_model.predict(predict)
        if(int(result) == 1):
            prediction = 'Sorry ! You have a chance of having Diabetes'
        else:
            prediction = 'Congrats ! you are Healthy'
    return(render_template("result.html", prediction=prediction))


@app.route('/cancer_result', methods=["POST"])
def cancer_result():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(float, predict_list))
        print(predict_list)
        predict = np.array(predict_list).reshape(1, 30)
        loaded_model = pickle.load(open("cancer_model.pkl", 'rb'))
        result = loaded_model.predict(predict)
        if(int(result) == 1):
            prediction = 'Sorry ! You have a chance of Breast Cancer'
        else:
            prediction = 'Congrats ! you are Healthy'
    return(render_template("result.html", prediction=prediction))


@app.route('/liver_result', methods=["POST"])
def liver_result():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(float, predict_list))
        print(predict_list)
        predict = np.array(predict_list).reshape(1, 10)
        loaded_model = pickle.load(open("liver_model.pkl", 'rb'))
        result = loaded_model.predict(predict)
        if(int(result) == 1):
            prediction = 'Sorry ! You have a chance of Liver Disease'
        else:
            prediction = 'Congrats ! you are Healthy'
    return(render_template("result.html", prediction=prediction))


@app.route('/heart_result', methods=["POST"])
def heart_result():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(float, predict_list))
        print(predict_list)
        predict = np.array(predict_list).reshape(1, 11)
        loaded_model = pickle.load(open("heart_model.pkl", 'rb'))
        result = loaded_model.predict(predict)
        if(int(result) == 1):
            prediction = 'Sorry ! You have a chance of Heart Attack'
        else:
            prediction = 'Congrats ! you are Healthy'
    return(render_template("result.html", prediction=prediction))


@app.route('/kidney_result', methods=["POST"])
def kidney_result():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(float, predict_list))
        print(predict_list)
        predict = np.array(predict_list).reshape(1, 13)
        loaded_model = pickle.load(open("kidney_model.pkl", 'rb'))
        result = loaded_model.predict(predict)
        if(int(result) == 1):
            prediction = 'Sorry ! You have a chance of Kidney Failure'
        else:
            prediction = 'Congrats ! you are Healthy'
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
