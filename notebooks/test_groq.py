# notebooks/test_groq.py
# Test de l'effet de la température sur les réponses de Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# Charger la clé depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvée dans .env")
    exit()

client = Groq(api_key=api_key)

# Prompt système et question utilisateur (ceux du Lab 5)
SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
         Tu recois un diagnostic et des donnees patient.
         Explique le resultat en francais simple,
         comme un medecin parlerait a son patient.
         Sois rassurant mais recommande une consultation.
         Maximum 3 phrases.
         Ne fais JAMAIS de diagnostic toi-meme."""

user_prompt = """Patient : Femme, 28 ans, region Dakar
         Symptomes : temperature 39.5, toux, fatigue, maux de tete
         Diagnostic du modele : paludisme (probabilite 72%)
         Explique ce resultat au patient."""

temperatures = [0.0, 0.5, 1.0]

for temp in temperatures:
    print(f"\n=== Température = {temp} ===")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=200,
        temperature=temp   # <-- c'est ici que ça change
    )
    print(response.choices[0].message.content)
    print(f"(Tokens utilisés : {response.usage.total_tokens})")