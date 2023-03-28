from flask import Flask, jsonify, render_template, request
from transformers import pipeline

app = Flask(__name__)

model_pipelines = {
    "opt-6.7": pipeline("text-generation", model="distilgpt2"),
    "alpaca-7": pipeline("text-generation", model="distilgpt2")
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chat():
    model_name = request.form["chatbot"]
    input_text = request.form["message"]
    print(input_text)
    print(model_name)

    if model_name and input_text:
        model = model_pipelines[model_name]
        response = model(input_text)[0]["generated_text"]
        print(response)
    else:
        response = "Something went wrong"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
