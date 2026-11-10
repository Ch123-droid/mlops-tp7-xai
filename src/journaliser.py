import hashlib
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from xai_commun import NOM, ALIAS, VERSION, m, Xte, yte


# Calcul de l'empreinte MD5 du fichier (meme principe que DVC)
def empreinte(chemin):
    return hashlib.md5(Path(chemin).read_bytes()).hexdigest()


# Predictions du modele
pred = m.predict(Xte)

# Creation de l'experience MLflow
mlflow.set_experiment('tp7-explicabilite')


# Enregistrement du run d'explicabilite
with mlflow.start_run(
    run_name=f'xai-{NOM}-v{VERSION.version}'
) as run:

    # Enregistrement des tags
    mlflow.set_tags({
        'modele': NOM,
        'alias': ALIAS,
        'modele_version': VERSION.version,
        'modele_run_id': VERSION.run_id,
        'donnees': 'data/transactions.csv',
        'donnees_md5': empreinte('data/transactions.csv'),
        'methodes': 'permutation ; SHAP TreeExplainer ; LIME'
    })

    # Enregistrement des parametres
    mlflow.log_params({
        'shap_echantillon': 400,
        'lime_num_samples': 2000,
        'lime_random_state': 0,
        'permutation_n_repeats': 10
    })

    # Enregistrement des metriques
    mlflow.log_metrics({
        'f1': f1_score(yte, pred),
        'precision': precision_score(yte, pred),
        'recall': recall_score(yte, pred)
    })

    # Archivage des graphiques, fichiers LIME et JSON
    for f in sorted(Path('reports').glob('*.*')):

        if f.suffix not in ('.png', '.html', '.json'):
            continue

        mlflow.log_artifact(
            str(f),
            artifact_path='explications'
        )

    # Sauvegarde de l'identifiant du run
    Path('reports/run_xai.txt').write_text(run.info.run_id)


# Association du run XAI a la version du modele
MlflowClient().set_model_version_tag(
    NOM,
    VERSION.version,
    'explications_run',
    run.info.run_id
)

print(
    'run', run.info.run_id[:8],
    ': explications du modele', NOM,
    'v' + VERSION.version,
    'archivees'
)