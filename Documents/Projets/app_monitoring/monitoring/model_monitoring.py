import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

def generate_monitoring_report():
    try:
        # Charge les données de référence
        reference_data = pd.read_csv("model/reference_data.csv")
        logger.info(f"Reference data loaded with columns: {reference_data.columns.tolist()}")
        
        # Charge les données de production
        current_data = pd.read_csv("model/predictions_log.csv")
        logger.info(f"Current data loaded with columns: {current_data.columns.tolist()}")

        # Assure que les colonnes correspondent
        required_columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
        
        # Vérifie et ajoute les colonnes target/prediction 
        for df in [reference_data, current_data]:
            if 'target' not in df.columns and 'prediction' in df.columns:
                df['target'] = df['prediction']
            elif 'prediction' not in df.columns and 'target' in df.columns:
                df['prediction'] = df['target']
            elif 'target' not in df.columns and 'prediction' not in df.columns:
                raise ValueError("Neither 'target' nor 'prediction' column found in data")

            # Vérifie les colonnes requises
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        # Crée le rapport avec seulement les métriques nécessaires
        report = Report(metrics=[
            DataDriftPreset(),
            ClassificationPreset()
        ])
        
        logger.info("Running report analysis...")
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Sauvegarde le rapport
        os.makedirs("monitoring/reports", exist_ok=True)
        report_path = f"monitoring/reports/model_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        
        logger.info(f"Report generated successfully at: {report_path}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating monitoring report: {str(e)}")
        raise