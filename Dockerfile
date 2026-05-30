# Dockerfile - SenSante
#image de base: python 3.11 leger
FROM python:3.11-slim
#Dossier de travail dans le conteneur
WORKDIR /app
#copier et installer les dépendances d'abord
#(optimisation du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#copier tout le code du projet
COPY . .
#Déclarer le port
EXPOSE 8000
#commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]