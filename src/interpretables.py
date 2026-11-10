import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from xai_commun import m, X, Xtr, Xte, ytr, yte, CAS

res = {}


# Fonction d'evaluation des modeles
def note(nom, modele, Xt):
    pred = modele.predict(Xt)

    res[nom] = {
        'f1': round(f1_score(yte, pred), 4),
        'fraudes_detectees': int(((pred == 1) & (yte == 1)).sum())
    }

    print(
        f'{nom:28s} F1 = {res[nom]["f1"]:.4f} | '
        f'fraudes detectees : {res[nom]["fraudes_detectees"]} / {int(yte.sum())}'
    )


# Transformation logarithmique du montant
def logm(d):
    d = d.copy()
    d['log_montant'] = np.log10(d.pop('montant'))
    return d


# 1. Evaluation de la foret aleatoire
note('foret fraude@production', m, Xte)


# 2. Regression logistique avec montant brut
lr_brut = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
).fit(Xtr, ytr)

note('logistique, montant brut', lr_brut, Xte)


# 3. Regression logistique avec log10(montant)
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
).fit(logm(Xtr), ytr)

note('logistique, log10(montant)', lr, logm(Xte))


# 4. Affichage des coefficients standardises
coef = {
    c: round(float(v), 2)
    for c, v in zip(logm(Xtr).columns, lr[-1].coef_[0])
}

print('coefficients standardises :', coef)


# 5. Calcul de l'effet sur les cotes
cote = np.exp(lr[-1].coef_[0] / lr[0].scale_)

print(
    'cote x', round(cote[0], 3),
    'par mois d\'anciennete ; x', round(cote[1], 2),
    'par transaction en 24 h'
)


# 6. Comparaison des arbres de decision
for p in [2, 3, 6, None]:

    t = DecisionTreeClassifier(
        max_depth=p,
        random_state=0
    ).fit(Xtr, ytr)

    note(
        f'arbre profondeur {p} ({t.get_n_leaves()} feuilles)',
        t,
        Xte
    )


# 7. Interpretation de l'arbre de profondeur 3
t3 = DecisionTreeClassifier(
    max_depth=3,
    random_state=0
).fit(Xtr, ytr)

print(
    export_text(
        t3,
        feature_names=list(X.columns),
        decimals=1
    )
)


# 8. Prediction pour Mme Benali
print(
    'Mme Benali : feuille/nœud',
    t3.apply(CAS)[0],
    '| P =',
    round(t3.predict_proba(CAS)[0, 1], 3)
)


# 9. Export des resultats au format JSON
with open('reports/interpretables.json', 'w') as f:
    json.dump(res, f, indent=2)

print('Resultats enregistres dans reports/interpretables.json')