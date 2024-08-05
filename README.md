# GPA Predictor
Our project aims to predict students' grades using an Artificial Neural Network (ANN). This is a type of machine learning model inspired by the way the human brain works. The ANN is trained to learn from data and make accurate predictions.

## How It Works:

Data Collection: First, we gather data about students. This data includes various features like previous grades, attendance records, participation in class activities, and other relevant factors that might influence a student's performance.

Data Preprocessing: Before feeding the data into our ANN, we clean and preprocess it. This step involves handling missing values, normalizing the data, and converting categorical data into numerical form.

Building the ANN Model: We design an ANN model with multiple layers. Each layer consists of neurons (nodes) that process input data. The connections between neurons have weights that are adjusted during the training process to improve prediction accuracy.

Training the Model: We train the ANN using the preprocessed data. During training, the model learns to recognize patterns and relationships between the features and the target variable (the grade). We use a portion of the data for training and another portion for validation to ensure the model generalizes well to new data.

Making Predictions: Once the model is trained, we use it to predict grades for new students based on their data. The model outputs a predicted grade, which can help in identifying students who might need additional support or resources.

Evaluation: We evaluate the model's performance using various metrics, such as  mean squared error and others. This helps us understand how well our model is performing and where improvements can be made.

## Benefits of the Project:

Early Intervention: By predicting grades, teachers and administrators can identify students who might be at risk of poor performance and provide timely support.
Personalized Learning: The model can help in creating personalized learning plans for students based on their predicted grades and areas of improvement.
Data-Driven Decisions: The use of an ANN provides a data-driven approach to understanding student performance, making educational strategies more effective.

## Requirements

- Python 3.12.+

## Installation

```bash
pip install -r requirements.txt
```

## Building Models

To build the models, run the Final.ipynb notebook. This will output the final GPA prediction model to a created `model`
folder along with essential utilities

## Running The Application

The application is hosted at [gpa-predictor.streamlit.app](https://gpa-predictor.streamlit.app/). You can also run the
application locally by running the following command:

```bash
streamlit run app.py
```

## YouTube link

https://youtu.be/ANWLOvSdrXk
