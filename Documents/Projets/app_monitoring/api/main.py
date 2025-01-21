from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from monitoring.model_monitoring import generate_monitoring_report
from sklearn.metrics import f1_score, accuracy_score
import logging
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application
app = FastAPI(title="Iris Prediction API with Monitoring")

class IrisInput(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

# Chargement du modèle
MODEL_PATH = "model/iris_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle non trouvé à {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    raise


# Collecte les métriques
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, include_in_schema=False)

# Crée un registre personnalisé
registry = CollectorRegistry()

# Défini des métriques
prediction_counter = Counter('iris_predictions_total', 'Total number of iris predictions', registry=registry)
prediction_duration = Histogram('iris_prediction_duration_seconds', 'Duration of iris predictions in seconds', registry=registry)
prediction_class_counter = Counter('iris_prediction_class', 'Count of predictions by class', 
                                 ['prediction_class'], registry=registry)
http_requests_total = Counter('http_requests_total', 'Total number of HTTP requests', ['method', 'endpoint'], registry=registry)
data_drift_score = Gauge('model_data_drift_score', 'Data drift score', registry=registry)
target_drift_score = Gauge('model_target_drift_score', 'Target drift score', registry=registry)
model_accuracy = Gauge('model_accuracy', 'Model accuracy score', registry=registry)
model_f1 = Gauge('model_f1_score', 'Model F1 score', registry=registry)

def update_monitoring_metrics(report):
    """Met à jour les métriques de monitoring."""
    try:
        # Métriques de dérive
        drift_metrics = report.metrics[0]
        data_drift_score.set(drift_metrics.get_score())
        
        performance_metrics = report.metrics[1]
        target_drift_score.set(performance_metrics.get_score())

        # Charge les données récentes pour calculer les performances
        recent_data = pd.read_csv("model/predictions_log.csv")
        if len(recent_data) > 0:
            y_true = recent_data['target'] if 'target' in recent_data else None
            y_pred = recent_data['prediction']
            
            if y_true is not None:
                # Mise à jour des métriques de performance
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                
                model_accuracy.set(accuracy)
                model_f1.set(f1)
                
                logger.info(f"Performance metrics updated - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    except Exception as e:
        logger.error(f"Error updating monitoring metrics: {str(e)}")

def save_prediction_data(input_df, prediction):
    try:
        log_path = "model/predictions_log.csv"
        
        # Prépare les données à sauvegarder
        save_df = input_df.copy()
        save_df['prediction'] = prediction
        save_df['target'] = prediction 
        
        # Crée le répertoire si nécessaire
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        if not os.path.exists(log_path):
            save_df.to_csv(log_path, index=False)
        else:
            save_df.to_csv(log_path, mode='a', header=False, index=False)
        
        logger.info(f"Prediction saved successfully: {save_df.to_dict('records')[0]}")
    except Exception as e:
        logger.error(f"Error saving prediction data: {str(e)}")
        raise

@app.post("/predict", response_model=dict)
async def predict(input_data: IrisInput):
    """Endpoint pour faire des prédictions."""
    http_requests_total.labels(method="POST", endpoint="/predict").inc()

    try:
        input_df = pd.DataFrame([{
            'sepal_length_cm': input_data.sepal_length_cm,
            'sepal_width_cm': input_data.sepal_width_cm,
            'petal_length_cm': input_data.petal_length_cm,
            'petal_width_cm': input_data.petal_width_cm
        }])

        with prediction_duration.time():
            prediction = model.predict(input_df)
            prediction_label = int(prediction[0])

        prediction_counter.inc()
        prediction_class_counter.labels(prediction_class=str(prediction_label)).inc()

        save_prediction_data(input_df, prediction_label)

        return {"prediction": prediction_label}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/monitoring/drift-report")
async def get_drift_report():
    """Génère et retourne un rapport de drift."""
    try:
        report = generate_monitoring_report()
        update_monitoring_metrics(report)
        
        return {
            "status": "success",
            "message": "Rapport généré avec succès",
            "metrics": {
                "data_drift_score": float(data_drift_score._value.get()),
                "target_drift_score": float(target_drift_score._value.get()),
                "model_accuracy": float(model_accuracy._value.get()),
                "model_f1": float(model_f1._value.get())
            }
        }
    except Exception as e:
        logger.error(f"Error generating drift report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Monte le dossier reports pour accès statique
app.mount("/reports", StaticFiles(directory="monitoring/reports"), name="reports")

@app.get("/reports/latest")
async def get_latest_report():
    """Retourne le dernier rapport généré."""
    reports_dir = "monitoring/reports"
    try:
        # Liste tous les rapports et prend le plus récent
        reports = sorted([f for f in os.listdir(reports_dir) if f.endswith('.html')])
        if not reports:
            raise HTTPException(status_code=404, detail="Aucun rapport disponible")
            
        latest_report = reports[-1]
        return FileResponse(
            path=os.path.join(reports_dir, latest_report),
            media_type='text/html'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)