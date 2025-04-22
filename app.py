from flask import Flask, request, render_template, Response
from langchain_community.llms import Ollama


app = Flask(__name__)

# Initialize LLM (No document retrieval, just pure LLM chat)
# Note: stream=True is the default for OllamaLLM so we don't need to specify it
llm = Ollama(model="llama3", temperature=2.0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return Response("Please enter a message.", mimetype='text/plain')
        
        # Define streaming generator function
        def generate():
            for chunk in llm.stream(user_input):
                # Each chunk is yielded as it comes from the LLM
                yield chunk
        
        # Use Flask's Response class with streaming
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        return Response(f"Error: {str(e)}", mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6060)