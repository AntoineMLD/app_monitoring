# API de Prédiction Iris avec Monitoring

Ce projet implémente une API FastAPI pour la prédiction des espèces d'Iris avec un système complet de monitoring incluant la détection de drift et le suivi des performances du modèle.

## 🌟 Fonctionnalités

- API REST pour les prédictions d'espèces d'Iris
- Monitoring des performances du modèle en temps réel
- Détection de drift des données via Evidently AI
- Métriques Prometheus
- Visualisation Grafana
- Logging détaillé
- Monitoring système via Node Exporter

## 🏗️ Architecture

Le projet utilise une architecture en microservices avec Docker Compose :
- `api`: Service FastAPI pour les prédictions
- `prometheus`: Collecte et stockage des métriques
- `grafana`: Visualisation des métriques
- `node_exporter`: Métriques système

## 🛠️ Prérequis

- Docker et Docker Compose
- Python 3.9+
- Git

## 📦 Installation

1. Cloner le repository :
```bash
git clone <repository-url>
cd app_monitoring
```

2. Créer les dossiers nécessaires :
```bash
mkdir -p model monitoring/reports
```

3. Préparation du modèle :
```python
# Exécuter le notebook de préparation du modèle
jupyter notebook model_preparation.ipynb
```

4. Construire et démarrer les services :
```bash
docker-compose up --build -d
```

## 🚀 Utilisation

### Faire des prédictions

```bash
# Exemple pour Iris Setosa (classe 0)
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2
}'

# Exemple pour Iris Versicolor (classe 1)
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
    "sepal_length_cm": 6.4,
    "sepal_width_cm": 2.9,
    "petal_length_cm": 4.3,
    "petal_width_cm": 1.3
}'

# Exemple pour Iris Virginica (classe 2)
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
    "sepal_length_cm": 6.3,
    "sepal_width_cm": 3.3,
    "petal_length_cm": 6.0,
    "petal_width_cm": 2.5
}'
```

### Accéder aux interfaces

- API Swagger: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)
- Rapports Evidently: http://localhost:8000/reports/latest

## 📊 Monitoring

### Métriques disponibles

- `iris_predictions_total`: Nombre total de prédictions
- `iris_prediction_duration_seconds`: Durée des prédictions
- `iris_prediction_class`: Compteur par classe prédite
- `model_data_drift_score`: Score de drift des données
- `model_accuracy`: Score d'exactitude du modèle
- `model_f1_score`: Score F1 du modèle

### Rapports de drift

Générer un nouveau rapport :
```bash
curl http://localhost:8000/monitoring/drift-report
```

Voir le dernier rapport :
```bash
curl http://localhost:8000/reports/latest
```

## 📁 Structure du Projet

```
.
├── api/
│   ├── Dockerfile
│   └── main.py
├── model/
│   ├── iris_model.pkl
│   └── reference_data.csv
├── monitoring/
│   ├── reports/
│   └── model_monitoring.py
├── prometheus/
│   └── prometheus.yml
├── docker-compose.yml
└── requirements.txt
```

## ⚙️ Configuration

### Prometheus

Le fichier `prometheus.yml` configure la collecte des métriques :
- Intervalle de scraping : 15s
- Endpoints monitorés :
  - API FastAPI (/metrics)
  - Node Exporter

Configuration :
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi-app'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'node'
    static_configs:
      - targets: ['node_exporter:9100']
```

### Grafana

- Port: 3001
- Credentials par défaut : admin/admin
- Source de données préconfigurée : Prometheus

![grafana02](https://github.com/user-attachments/assets/3fb1c4b4-1b54-45d2-8d59-35fb5bc7485a)
![grafana01](https://github.com/user-attachments/assets/b02450d4-bc43-49b9-a9a9-6ebc84a1ab26)



## 📝 Logging

Le système utilise le logging Python avec deux niveaux principaux :
- INFO: Opérations normales
- ERROR: Erreurs et exceptions

Les logs peuvent être consultés avec :
```bash
docker logs fastapi-api
```

## 🔄 Maintenance

### Sauvegarde des données
- Les prédictions sont sauvegardées dans `model/predictions_log.csv`
- Les rapports de drift dans `monitoring/reports/`

### Mise à jour du modèle
1. Entraîner un nouveau modèle
2. Sauvegarder dans `model/iris_model.pkl`
3. Redémarrer le service API :
```bash
docker-compose restart api
```

### Dépendances Python

Les dépendances principales sont :
```
evidently
pandas
scikit-learn
fastapi
uvicorn
joblib
pytest
prometheus_fastapi_instrumentator
psutil
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📜 License

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.
