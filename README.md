# cv5561-f25-team-MAH

Project Title: Live Feed age/gender/expression Tracking

Team Members:

Mohamed Khalil - khali178@umn.edu (Developer) (Coordinator)

Aaron Castle - castl145@umn.edu (Developer)

Hieu Le - le000515@umn.edu (Developer)

<<<<<<< HEAD
=======
# Environment setup

From the **repo root**:

```bash
# (recommended) create a venv
python -m venv .venv
# activate:
#   Windows: .venv\Scripts\activate
#   macOS/Linux: source .venv/bin/activate

# install deps for these training scripts
pip install -r requirements.txt
````

## How to run webpage

Ensure that all of the packages in the requirements.txt is installed

1. To run the flask server, execute: `python3 app.py`
2. Once the webserver is running, go to a webbrowser and type in: `localhost:5000`
3. Once on this page, the 2 models will be running and showing outputs. Any debugging statements can be seen in the terminal.

## Running individual models

For running individual models, you can navigate to the `individual_model_testing` folder where the files:

* faceDetection_yolo.py
* live_cnn_test.py

Are present in which they both can be ran by the command: `python3 <file_name>` which will start an OpenCV webcam displaying bounding boxes on faces with age, gender, and emotion predictions.

## Training models
Go to the `training_files` folder and follow the readme to train a model.
>>>>>>> website
