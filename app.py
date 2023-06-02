import os
from flask import Flask, render_template, request
from epilepsy_detection import perform_epilepsy_detection #calculate_accuracy

app = Flask(__name__)

# Load the pre-trained model
model_path = 'epilepsy_detection_model.h5'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'test_file' not in request.files:
            return render_template('index.html', message='Please select a test file')

        test_file = request.files['test_file']

        test_result = ""
        accuracy = None

        # Perform epilepsy detection on the uploaded test file
        if test_file:
            test_file_path = os.path.join('uploads', test_file.filename)
            test_file.save(test_file_path)
            test_result = perform_epilepsy_detection(model_path, test_file_path)

            # Calculate accuracy
            #accuracy = calculate_accuracy(model_path, test_file_path)

        # Render the result template with the testing result
        return render_template('result.html', test_result=test_result, accuracy=accuracy)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
