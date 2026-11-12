# Prédiction du désabonnement client télécom — MLOps

Mini-projet réalisé par Nada Kellouai et Khadija Chkhichkh à l’ENSA Fès.

## Objectif

Estimer le risque de désabonnement d’un client afin d’aider un opérateur
télécom à prioriser ses actions de fidélisation.

Le projet couvre la préparation des données, la comparaison de modèles,
le versionnement, le suivi des expériences, les contrôles de qualité,
une API locale et le monitoring.

## Données

Source : [Iranian Churn — UCI](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset).

Licence : CC BY 4.0. La source doit être citée lors de la réutilisation.

Le dataset contient 3 150 clients et 14 colonnes, dont la cible `Churn`.
Les caractéristiques sont agrégées sur les neuf premiers mois et le
désabonnement est observé à la fin du douzième mois.

Télécharger et décompresser le dataset, puis placer le fichier ici :

`data/raw/Customer Churn.csv`

Dix variables sont utilisées. La variable `age` est écartée car elle
correspond exactement à `age_group`. Les variables `status` et
`customer_value` sont écartées par précaution, faute de clarification
suffisante de leur définition.

Les profils identiques sont regroupés lors de la séparation :

| Ensemble | Clients | Taux de churn |
|---|---:|---:|
| Entraînement | 1 883 | 15,61 % |
| Validation | 632 | 15,82 % |
| Test | 635 | 15,91 % |

Aucun profil identique n’est partagé entre ces ensembles.

## Technologies et architecture

| Composant | Technologie |
|---|---|
| Préparation et modèles | Python, pandas, scikit-learn |
| Versionnement du code | Git et GitHub |
| Versionnement des données et pipeline | DVC |
| Expériences et registre de modèles | MLflow |
| API et validation | FastAPI et Pydantic |
| Tests | pytest |
| Intégration continue | GitHub Actions |

Le pipeline DVC prépare les données, entraîne les modèles et vérifie
les critères de qualité. Une commande distincte enregistre le candidat
dans MLflow. L’API charge une version précise avec son prétraitement
et applique le seuil de décision associé.

## Installation — Windows PowerShell

Environnement utilisé : Python 3.13.2.

```powershell
git clone https://github.com/NadaKellouai/telecom-churn-mlops.git
cd telecom-churn-mlops
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloque l’activation, autoriser les scripts pour cette
session uniquement :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Un dépôt privé nécessite une autorisation d’accès GitHub.

## Reproduction du pipeline

### 1. Vérifier les données

Après avoir placé le CSV téléchargé dans `data/raw/` :

```powershell
dvc status data/raw.dvc
```

La référence `data/raw.dvc` doit correspondre au contenu local.
Si DVC signale une différence, vérifier le fichier et le contenu du
dossier avant de poursuivre. Ne pas modifier la référence pour masquer
un écart.

### 2. Démarrer MLflow

Dans un premier terminal, à la racine du projet et avec `.venv` actif :

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Laisser ce terminal ouvert.

Interface : http://127.0.0.1:5000

### 3. Exécuter le pipeline

Dans un deuxième terminal avec `.venv` actif :

```powershell
dvc repro
```

Étapes :
- `prepare` : validation initiale et séparation des données ;
- `train` : entraînement et évaluation sur la validation ;
- `check_quality` : contrôle des métriques.

DVC ignore les étapes inchangées. Pour forcer une reconstruction :

```powershell
dvc repro --force
```

Une reconstruction crée de nouveaux runs MLflow. Les identifiants des
runs et certaines empreintes d’artefacts peuvent donc changer.

Le test n’est pas utilisé pour choisir le modèle ou le seuil.
Son évaluation initiale est disponible dans le notebook et MLflow.

### 4. Enregistrer le modèle

```powershell
python src/register_model.py
```

Cette commande :
- exécute le quality gate ;
- vérifie la cohérence du modèle local avec ses métriques ;
- sauvegarde et recharge le modèle dans MLflow ;
- vérifie la conservation des scores ;
- crée une version candidate avec son seuil.

Chaque exécution crée une nouvelle version. Relever le numéro affiché.

## Démarrer l’API

Dans un terminal libre, sélectionner le numéro de version affiché.
Exemple pour la version 2 :

```powershell
$env:MODEL_VERSION = "2"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Sur un registre neuf, la première version créée sera généralement 1.
La version par défaut de l’API est 1.

MLflow doit être accessible au démarrage de l’API.

- État : http://127.0.0.1:8000/health
- Interface interactive : http://127.0.0.1:8000/docs

Exemple de corps pour `POST /predict` :

```json
{
  "call_failure": 5,
  "complains": 1,
  "subscription_length": 34,
  "charge_amount": 1,
  "seconds_of_use": 1200,
  "frequency_of_use": 25,
  "frequency_of_sms": 10,
  "distinct_called_numbers": 10,
  "age_group": 3,
  "tariff_plan": 1
}
```

L’API renvoie le score, la décision, le seuil et la version.
Une valeur négative de `seconds_of_use` est refusée avec HTTP 422.

Le score n’est pas une certitude de départ ni une probabilité
dont la calibration aurait été démontrée.

## Résultats

Average precision sur la validation, métrique principale :

| Modèle | Average precision |
|---|---:|
| Baseline | 0,1582 |
| Régression logistique | 0,7937 |
| Random Forest | 0,9136 |

Le seuil 0,3 maximise le F1 de validation parmi les seuils
0,1 ; 0,2 ; 0,3 ; 0,4 ; 0,5 examinés.

Résultats initiaux du Random Forest sur le test au seuil 0,3 :

| Indicateur | Résultat |
|---|---:|
| Précision | 76,47 % |
| Rappel | 90,10 % |
| F1 | 0,8273 |
| Average precision | 0,8958 |
| Départs détectés | 91 sur 101 |
| Fausses alertes | 28 |

## Tests et quality gates

Tests d’intégration, avec MLflow accessible et une version enregistrée :

```powershell
python -m pytest tests/test_api.py -v
```

Tests du quality gate, sans serveur :

```powershell
python -m pytest tests/test_quality_gate.py -v
```

Contrôle des métriques :

```powershell
python src/check_quality.py
```

Le quality gate refuse :
- les métriques non finies ou hors de [0, 1] ;
- un candidat ne dépassant pas la baseline en average precision ;
- un candidat inférieur à la régression logistique sur cette métrique.

Ces critères évaluent le classement sur la validation.
Ils ne garantissent pas une rentabilité commerciale.

La CI GitHub Actions vérifie la syntaxe, les tests du quality gate et
les métriques de validation enregistrées. Elle ne réentraîne pas les
modèles et n’exécute pas les tests d’intégration MLflow.

## Monitoring

Les événements HTTP sont enregistrés dans `logs/requests.jsonl` :
horodatage, identifiant, route, statut, durée et version du modèle.
Les données clients ne sont pas enregistrées dans ces journaux.

```powershell
python src/monitor.py
```

Le script calcule les rejets de validation, les erreurs serveur,
la latence moyenne et le P95 des prédictions réussies.
Les quelques appels de démonstration ne constituent pas un benchmark.

Simulation de dérive :

```powershell
python src/check_drift.py
```

Le script compare une référence à un lot normal et à un lot dont
l’usage est volontairement réduit de 80 %. Il utilise uniquement
l’ensemble d’entraînement.

Une dérive déclenche une investigation ; elle ne prouve pas une
baisse de performance et ne déclenche pas de réentraînement automatique.

## Reproductibilité et limites

- Code et configurations : Git.
- Données et artefacts : références et cache DVC locaux.
- Environnement : `requirements.txt`.
- Séparation : graine 42 et manifeste des indices.
- Expériences : paramètres, métriques et modèles MLflow.

Aucun stockage distant DVC n’est configuré. Un clone Git ne contient
donc pas les CSV ni les modèles. La procédure retenue consiste à
récupérer le dataset public et reconstruire les artefacts.

Les étapes ont été exécutées sur l’environnement de développement ;
une reconstruction complète sur une deuxième machine n’a pas été testée.

Les données proviennent d’un seul opérateur. La séparation par profils
ne remplace pas une validation temporelle sur de nouvelles périodes.
Les performances hors ligne ne démontrent pas l’efficacité en production.

L’utilisation du groupe d’âge nécessite une analyse des biais.
Aucune équité du modèle ni efficacité causale des actions de
fidélisation n’a été démontrée.

Le déploiement est local. Docker et le déploiement cloud ne sont pas
utilisés.

## Démonstration de deux minutes

Montrer :
1. Le pipeline DVC et son quality gate.
2. Une expérience et une version dans MLflow.
3. Une prédiction valide et le rejet d’une entrée invalide.
4. La CI et les indicateurs de monitoring.

Mentionner au moins une limite : données d’un seul opérateur,
fausses alertes ou dérive sans preuve de baisse de performance.