from openrouter import OpenRouter
import os

# Clé API
or_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))

# Lister les modèles
models = or_client.models.list()

# Afficher les IDs valides
for model in models.data:  # <- note l'accès via .data et non ["data"]
    print(model.id)