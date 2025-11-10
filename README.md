Projet : Agent RAG – Résumé de documents de recettes
Objectif

Ce projet met en place un agent RAG (Retrieval-Augmented Generation) capable de lire un document PDF contenant des recettes de desserts, d’en extraire le contenu pertinent, puis de répondre à des questions ou de produire un résumé à partir des informations du document.

Technologies utilisées

Python 3.10+

LangChain – gestion du RAG et des outils

ChromaDB – stockage vectoriel

Ollama – modèles locaux (gemma2, mistral, etc.)

Google Gemini – modèle distant de chat

BM25 (rank-bm25) – recherche lexicale complémentaire

PyPDFium2 – extraction de texte depuis les PDF

Fonctionnalités principales

Extraction automatique du texte à partir d’un fichier PDF

Découpage du texte en chunks cohérents avec chevauchement

Indexation dans une base vectorielle (Chroma)

Recherche hybride combinant :

Recherche vectorielle (embeddings Nomic)

Recherche lexicale (BM25)

Agent intelligent capable :

d’utiliser un outil de recherche contextuelle

de reformuler les requêtes utilisateurs

de produire des réponses ancrées dans le contenu du document