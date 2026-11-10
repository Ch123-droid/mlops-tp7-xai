from lime.lime_tabular import LimeTabularExplainer

from xai_commun import m, X, Xtr, CAS


def expliquer(graine, n):
    ex = LimeTabularExplainer(
        Xtr.values,
        feature_names=list(X.columns),
        class_names=['normal', 'fraude'],
        mode='classification',
        random_state=graine
    )

    return ex.explain_instance(
        CAS.values[0],
        m.predict_proba,
        num_features=4,
        num_samples=n
    )


# Explication de reference
e = expliquer(0, 2000)

print('regles :', [
    (r, round(w, 3))
    for r, w in e.as_list()
])

# Sauvegarde de l'explication au format HTML
e.save_to_file('reports/lime_benali.html')

# Generation du graphique LIME
fig = e.as_pyplot_figure()
fig.tight_layout()
fig.savefig('reports/lime_benali.png', dpi=120)


# Test de stabilite selon la taille du voisinage
for n in [500, 2000]:

    print(f'--- {n} voisins montant nuit anciennete freq_24h')

    # 5 tirages aleatoires pour verifier la stabilite
    for graine in range(5):

        d = dict(expliquer(graine, n).as_list())

        w = [
            next(v for k, v in d.items() if col in k)
            for col in X.columns
        ]

        print(
            f'graine {graine} '
            f'{w[0]:+.3f} '
            f'{w[3]:+.3f} '
            f'{w[1]:+.3f} '
            f'{w[2]:+.3f}'
        )