from flask import Flask, jsonify, request
from flask_cors import CORS
from model_loader import model, image_loader, data_transforms

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict_sakura():
    # Return a BAD REQUEST response if there's no file in the request
    if request.files is None or len(request.files) < 1:
        return jsonify({'message': 'No file found'}), 400

    # Transform the request to a list of files with their key
    file_list = []
    for key, value in request.files.to_dict().items():
        file_list.append({'key': key, 'file_obj': value})

    # Collect all the outputs into a dictionary with key as the file key and value as the Sakura stage
    outputs = {}
    for file in file_list:
        output = model(image_loader(data_transforms, file['file_obj']))
        _, predicted = output.max(1)
        outputs[file['key']] = predicted.item()

    return jsonify({'results': outputs}), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)
