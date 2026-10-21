# FastAPI House Price Prediction

A FastAPI service that predicts California housing prices from the eight features used by the scikit-learn California Housing dataset. It also accepts a CSV upload and returns a CSV with predictions.

## Setup

Python 3.14 is the environment used for this project.

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The trained model is stored with Git LFS. Install Git LFS before cloning, then fetch the model file:

```powershell
git lfs install
git clone https://github.com/Thunder-67/fastapi-ml-prediction-api.git
cd fastapi-ml-prediction-api
git lfs pull
```

## Run the API

From the project directory with the virtual environment activated:

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

## Endpoints

- `GET /` - API welcome message.
- `GET /health` - Health and model information.
- `POST /predict` - Predict a price from one JSON object.
- `POST /predict_file` - Upload a CSV containing `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, and `Longitude`. The response is a CSV with predicted prices.

`predicted_prices.csv` contains five sample input rows.
