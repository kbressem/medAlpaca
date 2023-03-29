from flask import Flask, jsonify, render_template, request, Response
from transformers import pipeline
import time

# as alternative https://github.com/hyperonym/basaran

app = Flask(__name__)


# using pipelines does not work for streaming. 
# I probably need to implement the streaming directly with the model. 
# Given LLaMA is not fully supported by HF, this might even be the better solution
# TODO: use AutoTokenizer and AutoModel
# Encode the input, generate output. Length is n input tokes + output tokens. 
# Maybe look into this: https://huggingface.co/blog/how-to-generate

model_pipelines = {
    "opt-6.7": pipeline("text-generation", model="distilgpt2"),
    "alpaca-7": pipeline("text-generation", model="distilgpt2")
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def generate_response(model, input_text):
    response_parts = []
    generated_text = input_text
    timeout = 0

    while True:
        response_part = model(generated_text, max_length=10)[0]["generated_text"]
        response_parts.append(response_part)
        time.sleep(0.5)  # Simulate processing time
        generated_text += response_part
        yield f"data: {response_part.replace(input_text, '')}"

        if len(generated_text.replace(input_text, '')) > 2000 or timeout > 50:  # Limit the length of the generated text to prevent infinite loops
            break
        timeout += 1

    yield "data: END\n\n"


@app.route("/get_response", methods=["GET"])
def chat():
    model_name = request.args.get("chatbot")
    input_text = request.args.get("message")

    if model_name and input_text:
        model = model_pipelines[model_name]
        response = model(input_text)[0]["generated_text"]
        response = response.replace(input_text, "")
    else:
        response = "Something went wrong"
        return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
