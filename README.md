# Folder Structure:
├── app.py
├── train_model.py
├── data_processing.py
├── requirements.txt
├── Dockerfile
├── model/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
├── templates/
│   └── index.html
|---Database.db
         |---Heart_disease
# NOTE: 
This is Project structure have to folllow when you working on same project there is file which is not able to add in model folder  heart_disease_model.pkl due
to memory issue but after runing train_model.py file you get that file and for data which have to work that you i have added in above files ,you have to create 
"database.db" folder and add the file which i have added in above file "Heart_disease"

# 🫀 Heart Disease Prediction (Flask + ML)

This is a Flask-based machine learning app to predict heart disease using health-related attributes. It’s Docker-ready and includes model training, preprocessing, and a web form UI.

## Tech Stack

- Python 
- Flask
- Scikit-learn
- Docker
- HTML (Jinja2)

##  How to Run

# Option 1: Run Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py

--after above process done you can go for model deployment process

# Model_Deployment on AWS with EC2

# folder structure:
├── model/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
├── templates/
│   └── index.html
├── app.py
├── data_processing.py
├── requirements.txt
├── Dockerfile

# NOTE: Create a account on AWS and use above files for the process of model_deployment.
