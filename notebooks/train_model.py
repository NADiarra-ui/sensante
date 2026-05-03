import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques en nombres
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")     # (500,)

from sklearn.model_selection import train_test_split

# 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour le test
    random_state=42,    # reproductibilité
    stratify=y          # garder les mêmes proportions de diagnostics
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,    # 100 arbres de décision
    random_state=42      # reproductibilité
)

# Entraîner sur les données d'entraînement
model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions avec la réalité
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)

# Étape 5.1 : Comparaison des 10 premières prédictions
y_pred = model.predict(X_test)
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)

# Étape 5.2 : Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# Étape 5.3 : Matrice de confusion et rapport
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns

# Visualiser avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee dans figures/confusion_matrix.png")


import joblib
import os

# Créer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sérialiser le modèle
joblib.dump(model, "models/model.pkl")

# Vérifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder les encodeurs (indispensables pour les nouvelles données)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features (pour référence)
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes.")


# Charger le modèle DEPUIS LE FICHIER (pas depuis la mémoire)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")


# Un nouveau patient arrive au centre de santé de Médina
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder les valeurs catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
prediction = model_loaded.predict([features])[0]
probabilites = model_loaded.predict_proba([features])[0]

# Afficher le résultat
print(f"\nNouveau patient (Médina) :")
print(f"  Diagnostic prédit : {prediction}")
print("  Probabilités par classe :")
for classe, prob in zip(model_loaded.classes_, probabilites):
    print(f"    {classe:12s} : {prob:.2%}")


    # ========== EXERCICES ==========
print("\n=== Exercice 1 : Importance des features ===")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")


print("\n=== Exercice 2 : Tester avec d'autres patients ===")

# Patient 1 : jeune sans symptômes
sexe1 = le_sexe_loaded.transform(['M'])[0]
region1 = le_region_loaded.transform(['Dakar'])[0]   # région connue
feat1 = [20, sexe1, 37.0, 120, 0, 0, 0, region1]
diag1 = model_loaded.predict([feat1])[0]
print(f"Jeune sans symptômes (Dakar) -> {diag1}")

# Patient 2 : adulte avec forte fièvre
sexe2 = le_sexe_loaded.transform(['F'])[0]
region2 = le_region_loaded.transform(['Kaolack'])[0] # région connue
feat2 = [35, sexe2, 40.2, 100, 1, 1, 1, region2]
diag2 = model_loaded.predict([feat2])[0]
print(f"Adulte forte fièvre (Kaolack) -> {diag2}")

# Patient 3 : âgé avec toux
sexe3 = le_sexe_loaded.transform(['M'])[0]
region3 = le_region_loaded.transform(['Saint-Louis'])[0] # région connue
feat3 = [65, sexe3, 38.5, 140, 1, 0, 0, region3]
diag3 = model_loaded.predict([feat3])[0]
print(f"Âgé avec toux (Saint-Louis) -> {diag3}")