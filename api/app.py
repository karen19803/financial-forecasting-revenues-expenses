from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.persist import latest_model_for_department, load_model
from pathlib import Path
import pandas as pd

app = FastAPI(title='Forecast API')


class ForecastQuery(BaseModel):
    horizon: int = 3


MODEL_REGISTRY_DIR = 'model_registry'


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/forecast/{department}')
def get_forecast(department: str, horizon: int = 3):
    model_path = latest_model_for_department(MODEL_REGISTRY_DIR, department)
    if model_path is None:
        raise HTTPException(status_code=404, detail='No model found for department')
    # load model
    model = load_model(model_path)
    # build future dataframe
    # Prophet expects a pd.DataFrame with ds column
    last = model.history['ds'].max()
    last = pd.to_datetime(last)
    future = model.make_future_dataframe(periods=horizon, freq='M')
    forecast = model.predict(future)
    out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
    return out.to_dict(orient='records')
