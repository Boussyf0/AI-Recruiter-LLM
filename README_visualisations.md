# Visualisations des Résultats d'Évaluation

Ce dossier contient les visualisations générées à partir des résultats d'évaluation du système de matching CV-offres.

## Visualisations disponibles

1. **Précision de détection du domaine** (`domain_detection_accuracy.png`)
   - Graphique à barres montrant la précision globale de détection du domaine professionnel

2. **Matrice de confusion** (`confusion_matrix.png`)
   - Heatmap visualisant les prédictions correctes et incorrectes des domaines professionnels

3. **Métriques d'extraction des compétences** (`skills_metrics.png`)
   - Comparaison des scores de précision, rappel et F1 pour l'extraction des compétences

4. **Comparaison des compétences par CV** (`cv_skills_comparison.png`)
   - Graphique à barres groupées comparant le nombre de compétences attendues vs détectées pour chaque CV

5. **Distribution des scores de domaine** (`domain_radar_*.png`)
   - Graphiques radar montrant les scores attribués à chaque domaine pour un CV donné
   - Un graphique par CV testé

6. **Tableau de bord récapitulatif** (`evaluation_dashboard.png`)
   - Vue d'ensemble combinant les métriques principales et statistiques récapitulatives

## Comment interpréter les résultats

### Précision de détection du domaine
La valeur indique le pourcentage de CV pour lesquels le domaine principal a été correctement identifié.

### Matrice de confusion
- Les lignes représentent les domaines attendus (annotations manuelles)
- Les colonnes représentent les domaines prédits par le système
- Les valeurs sur la diagonale indiquent des prédictions correctes

### Métriques d'extraction des compétences
- **Précision**: Proportion des compétences extraites qui sont correctes
- **Rappel**: Proportion des compétences attendues qui ont été extraites
- **F1-Score**: Moyenne harmonique de la précision et du rappel

### Graphiques radar
Les graphiques radar montrent la distribution des scores pour chaque domaine. Le domaine attendu est indiqué par un marqueur "Attendu".

## Génération des visualisations

Pour régénérer ces visualisations avec de nouveaux résultats d'évaluation:

```bash
python visualize_evaluation.py --results-file data/evaluation/results/FICHIER_RESULTATS.json
```

Les visualisations seront sauvegardées dans le dossier `data/evaluation/visualizations/`. 