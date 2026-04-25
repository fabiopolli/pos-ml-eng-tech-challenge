from fastapi import APIRouter
from app.schemas.input import InputData
from app.services.model_service import predict

router = APIRouter()

@router.post("/predict")
def predict_route(data: InputData):
    prediction = predict(data)

    return {
        "prediction": int(prediction[0])
    }