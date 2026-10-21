from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import joblib
import io
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse


app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


class Housefeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in block group")
    HouseAge: float = Field(gt=0, description="Average house age in block group")
    AveRooms: float = Field(gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per household")
    Population: float = Field(gt=0, description="Total number of people residing in the block group")
    AveOccup: float = Field(gt=0, description="Average number of occupants per household")
    Latitude: float = Field(gt=32, lt=42, description="Latitude of the block group")
    Longitude: float = Field(gt=-125, lt=-114, description="Longitude of the block group")

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"
            , "status": "running",
            "endpoint": "send a POST request to /predict with the following JSON body to get a prediction",
            
        }
@app.get("/health")
def health():
    return {"status": "running",
            "model": "RandomForestRegressor",
            "features": features,
            "avg_error": "$39000"
            }
@app.post("/predict")
def predict_price(house: Housefeatures):
    try:
        input_data = pd.DataFrame([
            {
                "MedInc": house.MedInc,
                "HouseAge": house.HouseAge,
                "AveRooms": house.AveRooms,
                "AveBedrms": house.AveBedrms,
                "Population": house.Population,
                "AveOccup": house.AveOccup,
                "Latitude": house.Latitude,
                "Longitude": house.Longitude
            }
        ])
        prediction = model.predict(input_data)
        price_usd = prediction[0] * 100000  # Convert to USD
        return {"predicted_price": price_usd,
                "predicted_price_short": f"${price_usd:,.0f} hundreds of thousands",
                "fidence_range": f"${price_usd - 39000:,.0f} - ${price_usd + 39000:,.0f}"
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_file")
async def predict_price_file(file: UploadFile = File(...)):
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, 
                detail="Invalid file format. Please upload a CSV file."
            )
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        if len(df) == 0:
            raise HTTPException(status_code=400, 
                detail="The uploaded CSV file is empty."
            )
        try:
            predictions = model.predict(df[required_columns])
            df["predicted_price"] = predictions * 100000  # Convert to USD
            df["predicted_price_short"] = df["predicted_price"].apply(lambda x: f"${x:,.0f} hundreds of thousands")
            df["confidence_range"] = df["predicted_price"].apply(lambda x: f"${x - 39000:,.0f} - ${x + 39000:,.0f}")

            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")