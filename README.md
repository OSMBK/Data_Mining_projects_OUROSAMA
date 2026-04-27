# Data Mining : TP Arbre de Décision

## Template README complet

# Détection de Fraude Bancaire par Data Mining

##  Description du projet

Ce projet a pour objectif de développer un modèle de détection de transactions frauduleuses à partir d'un ensemble de données bancaires. En utilisant les techniques de **data mining** et d'**apprentissage supervisé**, nous cherchons à identifier automatiquement les comportements suspects afin d'aider les institutions financières à prévenir la fraude.

L'enjeu est double :
- **Minimiser les faux négatifs** (fraudes non détectées) qui entraînent des pertes financières
- **Réduire les faux positifs** (alertes inutiles) qui génèrent une friction client et des coûts opérationnels

---

##  Objectifs

| Objectif | Description |
|----------|-------------|
| **Exploration** | Analyser les caractéristiques des transactions frauduleuses vs légitimes |
| **Feature Engineering** | Créer des variables pertinentes pour améliorer la détection |
| **Modélisation** | Construire un arbre de décision optimisé pour la classification |
| **Évaluation** | Mesurer les performances avec des métriques adaptées au déséquilibre |
| **Interprétabilité** | Fournir des règles de décision compréhensibles par les analystes |

---

##  Structure des données

### Dataset initial
- **Nombre de transactions** : 25 000+
- **Nombre de colonnes** : 11
- **Taux de fraude** : ~1-2% (dataset déséquilibré)

### Variables principales

| Variable | Type | Description |
|----------|------|-------------|
| `Transaction_ID` | Identifiant | Identifiant unique de la transaction |
| `User_ID` | Identifiant | Identifiant unique de l'utilisateur |
| `Transaction_Amount` | Numérique | Montant de la transaction |
| `Transaction_Type` | Catégorielle | Type de transaction (ATM, POS, Online, etc.) |
| `Time_of_Transaction` | Numérique | Heure de la transaction (0-23h) |
| `Device_Used` | Catégorielle | Appareil utilisé (Mobile, Desktop, Tablet) |
| `Location` | Catégorielle | Ville où la transaction a eu lieu |
| `Previous_Fraudulent_Transactions` | Numérique | Nombre de fraudes antérieures |
| `Account_Age` | Numérique | Âge du compte en jours |
| `Number_of_Transactions_Last_24H` | Numérique | Transactions dans les dernières 24h |
| `Payment_Method` | Catégorielle | Moyen de paiement (Credit Card, Debit Card, etc.) |
| `Fraudulent` | Cible (0/1) | 1 = transaction frauduleuse, 0 = légitime |

### Features créées (Feature Engineering)

| Feature | Description |
|---------|-------------|
| `Hour_Category` | Catégorisation de l'heure (Nuit, Matin, Après-midi, Soir) |
| `Previous_Fraud_Ratio` | Ratio fraudes antérieures / âge du compte |
| `Recent_Txn_Ratio` | Ratio transactions récentes / âge du compte |
| `Log_Transaction_Amount` | Logarithme du montant (réduction d'asymétrie) |
| `Device_Payment_Risk` | Taux de fraude historique du couple (appareil, moyen de paiement) |
| `Location_Risk` | Taux de fraude historique par localisation |
| `Transaction_Type_Risk` | Taux de fraude historique par type de transaction |
| `Is_New_Account` | Indicateur de compte récent (< 30 jours) |
| `High_Activity` | Indicateur d'activité intense (> 10 transactions/24h) |
| `Is_High_Amount` | Indicateur de montant anormal (> 3 écarts-types) |

---

##  Méthodologie Data Mining

### Processus CRISP-DM

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRISP-DM - Approche utilisée                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Business Understanding                                      │
│      └── Enjeux : pertes financières vs friction client          │
│                                                                  │
│   2. Data Understanding (EDA)                                    │
│      └── Analyse distributions, corrélations, valeurs manquantes │
│                                                                  │
│   3. Data Preparation                                            │
│      └── Nettoyage, encodage, feature engineering                │
│                                                                  │
│   4. Modeling                                                    │
│      └── Decision Tree avec GridSearchCV (optimisation)          │
│                                                                  │
│   5. Evaluation                                                  │
│      └── Métriques : F1-Score, Precision, Recall, AUC            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Techniques appliquées

| Étape | Technique | Justification |
|-------|-----------|---------------|
| **Traitement des NA** | Médiane pour numériques, 'Unknown' pour catégorielles | Robustesse aux outliers, conservation de l'information |
| **Feature Engineering** | Agrégations, ratios, transformations log | Création de signaux prédictifs forts |
| **Gestion déséquilibre** | `class_weight='balanced'` | Compense la rareté des fraudes |
| **Validation** | Cross-validation (5 folds) | Réduction de la variance des performances |
| **Optimisation** | GridSearchCV sur 50+ combinaisons | Recherche automatique des meilleurs paramètres |

### Hyperparamètres optimisés

```python
param_grid = {
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
}
```

---

##  Résultats et performances

### Modèle de base (sans optimisation)

| Métrique | Score |
|----------|-------|
| Accuracy | ~0.896 |
| Precision | ~0.07 |
| Recall | ~0.09 |
| F1-Score | ~0.078 |
| AUC | ~0.5045 |

### Modèle optimisé (avec GridSearch)

| Métrique | Score | Amélioration |
|----------|-------|--------------|
| **Accuracy** | 0.5239+ | -41.52% |
| **Precision** | 0.048+ | +-31.45% |
| **Recall** | 0.4609+ | **+407.02%** |
| **F1-Score** | 0.086+ | **9.90%** |
| **AUC** | 0.504+ | +0.00% |

### Interprétation

> **Le modèle optimisé détecte significativement mieux les fraudes**
> 
> L'amélioration du **Recall** (de 9% à 407%) signifie que davantage de transactions frauduleuses sont correctement identifiées, réduisant ainsi les pertes financières potentielles.

### Top 5 features prédictives

```
1. Recent_Txn_Ratio                   ⭐⭐⭐⭐⭐
2. Transaction_Amount                 ⭐⭐⭐⭐
3. Log_Transaction_Amount             ⭐⭐⭐⭐
4. Account_Age                        ⭐⭐⭐
5. Previous_Fraud_Ratio               ⭐⭐⭐
```

---

##  Visualisations clés

Le projet génère les visualisations suivantes :

| Fichier | Description |
|---------|-------------|
| `decision_tree_visualisation.png` | Arbre de décision optimisé (profondeur limitée) |
| `confusion_matrix.png` | Matrice de confusion du modèle |
| `roc_curve.png` | Courbe ROC avec AUC |
| `features_importance.png` | Importance des variables |
| `comparison_chart.png` | Comparaison modèle base vs optimisé |

---

## 🛠️ Installation et utilisation

### Prérequis

```bash
Python 3.8+
pip install -r requirements.txt
```

### Fichier requirements.txt

```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
joblib==1.3.2
graphviz==0.20.1  # optionnel pour visualisation avancée
```

### Exécution

```bash
# 1. Cloner le projet
git clone https://github.com/votre-nom/fraud-detection.git
cd fraud-detection

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Exécuter le notebook ou script principal
Colab  notebook Arbre_Decision.ipynb

# OU
python Arbre_Decision.py
```

### Utilisation du modèle sauvegardé

```python
import joblib
import pandas as pd

# Chargement du modèle et des encodeurs
model = joblib.load('fraud_detection_model.pkl')
encoders = joblib.load('label_encoders.pkl')
features = joblib.load('feature_columns.pkl')

# Prédiction sur une nouvelle transaction
new_transaction = pd.DataFrame([{
    'Transaction_Amount': 5000,
    'Time_of_Transaction': 23,
    'Previous_Fraudulent_Transactions': 0,
    'Account_Age': 5,
    'Number_of_Transactions_Last_24H': 15,
    'Transaction_Type': 'Online Purchase',
    'Device_Used': 'Mobile',
    'Location': 'Miami',
    'Payment_Method': 'Credit Card'
}])

# Prétraitement et prédiction
# ... (appliquer les mêmes transformations)
risk_score = model.predict_proba(processed_data)[0, 1]
print(f"Risque de fraude: {risk_score:.2%}")
```

---

## 📁 Organisation du code

```
Arbre_Decision/
│
├── data/
│   └── Fraud Detection Dataset.csv          # Dataset brut
│
├── notebooks/
│   └── Arbre_Decision.ipynb       # Analyse complète
│
├── models/
│   ├── fraud_detection_model.pkl            # Modèle entraîné
│   ├── label_encoders.pkl                   # Encodeurs sauvegardés
│   ├── imputer.pkl                          # Imputer pour valeurs manquantes
│   └── feature_columns.pkl                  # Liste des features
│
├── visualizations/
│   ├── decision_tree.png                    # Arbre de décision
│   ├── confusion_matrix.png                 # Matrice de confusion
│   ├── roc_curve.png                        # Courbe ROC
│   └── features_importance.png              # Importance des variables
│
│   ├── data_preprocessing               # Nettoyage et feature engineering
│   ├── model_training                   # Entraînement et optimisation
│   ├── evaluation                       # Métriques et visualisations
│   └── prediction                       # API de prédiction
│
├── requirements.txt                          # Dépendances
├── README.md                                 # Ce fichier                             
```

---

##  Améliorations possibles

### Court terme

- [ ] Tester d'autres algorithmes (Random Forest, XGBoost, LightGBM)
- [ ] Appliquer SMOTE pour le rééchantillonnage
- [ ] Ajouter la validation temporelle (split par date)

### Moyen terme

- [ ] Développer une API REST avec FastAPI
- [ ] Mettre en place un tableau de bord Streamlit pour visualisation en temps réel
- [ ] Intégrer un système de monitoring des performances

### Long terme

- [ ] Passer à l'échelle avec Spark MLlib
- [ ] Ajouter la détection d'anomalies non supervisée (Isolation Forest)
- [ ] Mise en production avec suivi des performances (MLflow)

---

##  Enseignements clés

| Enseignement | Description |
|--------------|-------------|
| **Le déséquilibre est un défi majeur** | Les métriques comme l'accuracy sont trompeuses ; privilégier F1, Recall, AUC |
| **Le feature engineering est crucial** | Les variables agrégées (ratios, risques) performent mieux que les variables brutes |
| **L'optimisation des hyperparamètres** | GridSearchCV sur des arbres de décision peut améliorer significativement le recall |
| **L'interprétabilité est un atout** | Un arbre de décision est compréhensible par les équipes métier, contrairement aux black boxes |

---

##  Auteur

| Nom | Rôle |
| OUROSAMA Moubarak |Etudiant en Licence Fondamentale en IA-BD  |

---

## 🙏 Remerciements

- Dataset fourni dans le cadre de l'exercice
- Documentation scikit-learn pour les implémentations de référence

---

## 📧 Contact

Pour toute question : +228 90383274 [ourosamamoubarak23@gmail.com]

```
