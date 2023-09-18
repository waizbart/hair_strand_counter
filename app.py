from flask import Flask, request, jsonify
from counter import HairCounter
from PIL import Image
import numpy

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return '<h1>Hello from hair strand counter!</h1>'

@app.route("/count_strands", methods=["POST"])
def process_img():
    file = request.files['image']

    img = numpy.array(Image.open(file.stream))
    
    hair_counter = HairCounter(img)
    hair_counter.run()

    return jsonify({'message': 'success', 'strand_number': hair_counter.hair_number, 'used_area': hair_counter.used_area})


if __name__ == "__main__":
    app.run(port=3403)