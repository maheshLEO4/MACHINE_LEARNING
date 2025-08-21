# 🏡 House Price Prediction App

A Flask-based web application and REST API to predict house prices using a trained ML model.  
Includes a `Dockerfile` for containerized deployment.

---

## ✅ Features
- **Web Interface**: Simple form to input details and get predictions.
- **REST API**: `/predict` endpoint for programmatic access.
- **Machine Learning**: Uses pre-trained model and encoder.
- **Dockerized**: Easy to deploy anywhere.

---

## 🖥️ Project Structure
```
.
├── app.py                # Flask backend
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image setup
├── models/               # ML model & encoder files
│   ├── house_price_model.pkl
│   └── encoder.pkl
└── templates/
    ├── index.html # Frontend page
    └── static/
        ├── style.css
        └── script.js
```

---

## ▶️ Run Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run Flask app:
```bash
python app.py
```

Now open [http://localhost:5000](http://localhost:5000) in your browser for the **frontend**.

---

## 🐳 Run with Docker

Build the image:
```bash
docker build -t house-price-app .
```

Run the container:
```bash
docker run -p 5000:5000 house-price-app
```

---

## 🌐 API Endpoint

**POST** `/predict`  

### Request Body:
```json
{
  "title": "3 BHK",
  "location": "Hyderabad",
  "area": 1200
}
```

### Response:
```json
{
  "title": "3 BHK",
  "location": "Hyderabad",
  "area_insqft": 1200,
  "predicted_price(L)": 85.67
}
```

---

## ✨ Frontend Screenshot
```
![UI](image.png)
```

---




