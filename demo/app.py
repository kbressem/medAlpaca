from flask import Flask, jsonify, render_template, request, Response
from transformers import pipeline
import time

app = Flask(__name__)

model_pipelines = {
    "opt-6.7": pipeline("text-generation", model="distilgpt2"),
    "alpaca-7": pipeline("text-generation", model="distilgpt2")
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def generate_response(model, input_text):
    response_parts = []

    for i in range(1, 4):  # Arbitrary number of response parts
        response_part = model(input_text, max_length=i * 10)[0]["generated_text"]
        response_parts.append(response_part)
        yield f"data: {response_part}\n\n"
        time.sleep(1)  # Simulate processing time

    yield "data: END\n\n"


@app.route("/get_response", methods=["GET"])
def chat():
    model_name = request.args.get("chatbot")
    input_text = request.args.get("message")

    if model_name and input_text:
        model = model_pipelines[model_name]
        return Response(generate_response(model, input_text), content_type="text/event-stream")
    else:
        response = "Something went wrong"
        return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
