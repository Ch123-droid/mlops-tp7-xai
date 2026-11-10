import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import shap

from xai_commun import m, Xte, CAS

# Creation de l'explicateur SHAP
ex = shap.TreeExplainer(m)

# Calcul des probabilites de fraude
proba = m.predict_proba(Xte)[:, 1]

# Identification de la transaction de test la plus suspecte
idx = Xte.index[proba.argmax()]

# Explication des deux transactions
for nom, x in [('benali', CAS), (f'test_{idx}', Xte.loc[[idx]])]:

    # Calcul des valeurs SHAP pour la classe fraude
    e = ex(x)[0, :, 1]

    print(f'--- {nom} :', x.iloc[0].to_dict())
    print('valeur de base :', round(float(e.base_values), 4))

    # Affichage des contributions de chaque variable
    for col, c in sorted(
        zip(x.columns, e.values),
        key=lambda t: -abs(t[1])
    ):
        print(f' {col:11s} {c:+.4f}')

    # Verification de la prediction
    print(
        'base + somme :',
        round(float(e.base_values + e.values.sum()), 4),
        '| prediction :',
        round(m.predict_proba(x)[0, 1], 4)
    )

    # Generation du graphique Waterfall
    shap.plots.waterfall(e, show=False)

    plt.tight_layout()
    plt.savefig(f'reports/shap_{nom}.png', dpi=120)
    plt.close('all')