## ML Deployment Architecture

This project demonstrates how to deploy a machine learning model using **Django** as a web application framework.

The system separates the **user interface**, **web logic**, and **machine learning inference layer**, following a common pattern used in production ML services.

```
Browser UI
   ↓
Django Views
   ↓
Prediction Service
   ↓
Serialized ML Model (.pkl)
```

### Architecture Explanation

**1. Browser UI**

The user interacts with the application through a web form where they input Iris flower measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The form is rendered using Django templates and styled with Bootstrap.

---

**2. Django Views**

Django views handle HTTP requests and responses:

* `home` view displays the input form.
* `predict` view processes form submissions and calls the ML prediction service.

Views act as the bridge between the **user interface** and the **machine learning layer**.

---

**3. Prediction Service**

The prediction service contains the core inference logic. It:

* loads the trained model
* formats user input into the correct feature structure
* generates predictions

Separating inference logic from views improves maintainability and mirrors real-world ML system design.

---

**4. Serialized Machine Learning Model**

The trained model is stored as a serialized artifact:

```
iris_model.pkl
```

The model is trained using **Scikit-learn's RandomForestClassifier** on the Iris dataset and loaded by the application for real-time predictions.

---

### Project Structure

```
lab-django-ds-deployment
│
├── manage.py
├── requirements.txt
│
├── ml_project
│   └── Django project configuration
│
└── predictor
    │
    ├── model
    │   ├── train.py
    │   └── iris_model.pkl
    │
    ├── services
    │   └── inference logic
    │
    ├── templates
    │   └── HTML interface
    │
    ├── views.py
    └── urls.py
```

This modular structure reflects common **ML engineering deployment patterns**, where training, inference, and web layers are clearly separated.

------------------------------------------------------
## System Overview

The following diagram illustrates the flow of data through the deployed machine learning application.

```
User Input (Browser Form)
        │
        ▼
Django Web Server
        │
        ▼
Prediction View
        │
        ▼
Inference Service
        │
        ▼
Serialized Model (iris_model.pkl)
        │
        ▼
Prediction Result Returned to Browser
```

This architecture separates the **presentation layer**, **application logic**, and **machine learning inference layer**, which is a common design pattern used in production ML systems.

---

## Example Prediction

Example input submitted through the web interface:

```
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

Model prediction:

```
Predicted Iris Species: Setosa
```

The prediction is generated using a trained **Random Forest Classifier**.

---

## Model Details

| Attribute  | Value                     |
| ---------- | ------------------------- |
| Dataset    | Iris Dataset              |
| Task       | Multiclass Classification |
| Algorithm  | Random Forest             |
| Framework  | Scikit-learn              |
| Deployment | Django Web Application    |

---

## Model Performance

The model was trained on the classic Iris dataset and achieves approximately:

```
Accuracy ≈ 95–98%
```

This performance is typical for tree-based models trained on the Iris dataset.

---

## Technologies Used

* **Python**
* **Django**
* **Scikit-learn**
* **NumPy**
* **Pandas**
* **Joblib**
* **Bootstrap**

---

## Running the Application Locally

Start the Django server:

```
python manage.py runserver
```

Then open your browser and navigate to:

```
http://127.0.0.1:8000
```

Enter Iris flower measurements to generate predictions in real time.
