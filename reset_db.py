#!/usr/bin/env python
import httpx
import asyncio
import argparse
from typing import List

# Configuration par défaut
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTIONS = ["cvs", "jobs"]

async def reset_collections(qdrant_url: str, collections: List[str]):
    """Réinitialiser les collections dans Qdrant"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for collection in collections:
            print(f"Suppression de la collection '{collection}'...")
            try:
                # Vérifier si la collection existe
                response = await client.get(f"{qdrant_url}/collections/{collection}")
                if response.status_code == 200:
                    # Supprimer la collection si elle existe
                    delete_response = await client.delete(f"{qdrant_url}/collections/{collection}")
                    if delete_response.status_code == 200:
                        print(f"✅ Collection '{collection}' supprimée avec succès")
                    else:
                        print(f"❌ Erreur lors de la suppression de la collection '{collection}': {delete_response.status_code}")
                        print(f"Détails: {delete_response.text}")
                elif response.status_code == 404:
                    print(f"⚠️ La collection '{collection}' n'existe pas, rien à faire")
                else:
                    print(f"❌ Erreur lors de la vérification de la collection '{collection}': {response.status_code}")
                    print(f"Détails: {response.text}")
            except Exception as e:
                print(f"❌ Exception lors de la suppression de la collection '{collection}': {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Réinitialiser les collections dans Qdrant")
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL, help="URL de l'API Qdrant")
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS, help="Collections à réinitialiser")
    
    args = parser.parse_args()
    
    # Exécuter la tâche asynchrone
    asyncio.run(reset_collections(args.url, args.collections))
    print("\nRéinitialisation des collections terminée!")

if __name__ == "__main__":
    main() 