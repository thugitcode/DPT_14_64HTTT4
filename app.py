import os
import pickle
from flask import Flask, request, render_template, url_for
from src.search import search

app = Flask(__name__)

# Load index đã lưu
with open("features_index.pkl", "rb") as f:
    index = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["query"]
        filename = file.filename
        query_path = os.path.join("static/queries", filename)
        file.save(query_path)

        results = search(query_path, index, topk=5)
        return render_template("results.html", query=filename, results=results)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
