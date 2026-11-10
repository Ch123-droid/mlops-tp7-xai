import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from xai_commun import m, Xte

# Selection de 400 transactions de test
Xs = Xte.sample(400, random_state=0)

# Creation de l'explicateur SHAP
ex = shap.TreeExplainer(m)

# Calcul des valeurs SHAP pour la classe 1 (fraude)
sv = ex(Xs)[:, :, 1]

# Affichage de la valeur de base
print('valeur de base :', round(float(sv.base_values[0]), 4))

# Calcul de l'importance globale des variables
imp = pd.Series(
    np.abs(sv.values).mean(0),
    index=Xs.columns
)

print(imp.sort_values(ascending=False).round(4).to_string())

# Export des importances SHAP
imp.round(4).to_json('reports/shap_importance.json')

# Generation du graphique SHAP global
shap.plots.beeswarm(sv, show=False)

plt.tight_layout()
plt.savefig('reports/shap_global.png', dpi=120)
plt.close('all')

# Visualisation de l'impact du montant
plt.scatter(
    Xs['montant'],
    sv[:, 'montant'].values,
    c=Xs['nuit'],
    cmap='coolwarm',
    s=12
)

plt.xscale('log')
plt.xlabel('montant (MAD)')
plt.ylabel('contribution SHAP du montant')
plt.colorbar(label='nuit')
plt.tight_layout()

plt.savefig('reports/shap_montant.png', dpi=120)

print('graphiques ecrits')