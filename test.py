import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime as dt



from sklearn.ensemble import RandomForestRegressor  # Modèle de régression par forêts aléatoires
from sklearn.linear_model import LinearRegression   # Modèle de régression linéaire
from sklearn.svm import SVR                         # Modèle de régression basé sur les machines à vecteurs de support (SVM)
from sklearn.preprocessing import MinMaxScaler      # Outil pour normaliser les données
from sklearn.metrics import mean_squared_error, r2_score  # Métriques pour évaluer les performances des modèles
from datetime import datetime as dt                # Bibliothèque pour travailler avec les dates
# Importation des bibliothèques nécessaires pour les analyses
from sklearn.preprocessing import StandardScaler  # Pour la normalisation des données
from sklearn.decomposition import PCA  # Pour la réduction de la dimensionnalité
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de test
from sklearn.linear_model import LogisticRegression  # Pour le modèle de régression logistique
from sklearn.metrics import classification_report  # Pour évaluer les performances du modèle
# Titre principal
st.markdown("# 📊 Tableau de bord de la prédiction de la durée de vie des moteurs")
st.markdown("### Analyse de la performance des moteurs Turbofan à partir de données de capteurs")

# Ajout d'une image (optionnelle) - vous pouvez ajouter votre logo ici
# st.image('path_to_logo.png', width=200)

# Menu Latéral
st.sidebar.title("Navigation")
menu_options = ["Aperçu des données", "Analyse des capteurs", "PCA et Régression Logistique", "Modèles de Prédiction"]
selection = st.sidebar.radio("Choisir une section", menu_options)

# Chargement des données
@st.cache_data
def load_data():
    columns = ['engine', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
    train_data = pd.read_csv("train_FD001.txt", sep='\\s+', names=columns)
    return train_data

data = load_data()

# Affichage des premières lignes
st.subheader("Aperçu des données")
st.dataframe(data.head())

# Ajout de la colonne RUL
def add_remaining_RUL(data):
    max_cycles = data.groupby('engine')['cycle'].max()
    merged = data.merge(max_cycles.to_frame(name='max_cycles'), left_on='engine', right_index=True)
    merged["RUL"] = merged["max_cycles"] - merged['cycle']
    return merged.drop("max_cycles", axis=1)

data = add_remaining_RUL(data)

# Sélection d'un moteur
engine_selected = st.selectbox("Sélectionnez un moteur", data["engine"].unique())
engine_data = data[data["engine"] == engine_selected]

# Recherche du nombre maximum de cycles pour chaque moteur
def find_max_cycle(data):
    st.subheader("Nombre maximum de cycles pour chaque moteur")
    max_cycle = data[['engine', 'cycle']].groupby(['engine']).count().reset_index().rename(columns={'cycle': 'max_cycles'})
    return max_cycle

max_cycle = find_max_cycle(data)
st.dataframe(max_cycle)

# Affichage des graphiques
st.subheader("Évolution des capteurs et réglages au fil des cycles")
fig, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=True)
axes = axes.flatten()
columns_to_plot = ['setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 16)]

for i, col in enumerate(columns_to_plot):
    axes[i].plot(engine_data['cycle'], engine_data[col], label=col)
    axes[i].set_title(col)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
st.pyplot(fig)

# Affichage du graphique des cycles maximums
def barplt(data):
    fig, ax = plt.subplots(figsize=(15,10))
    sns.barplot(x='engine', y='max_cycles', data=data, palette='coolwarm', ax=ax)
    sns.set_context("notebook")
    ax.set_title('Durée de vie des moteurs Turbofan', fontweight='bold', fontsize=20)
    ax.set_xlabel('Moteurs', fontweight='bold', fontsize=15)
    ax.set_ylabel('Cycles maximum', fontweight='bold', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.grid(visible=True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

barplt(max_cycle)

# Analyse des relations entre deux variables (nuages de points)
st.subheader("Relation entre op_setting_1 et op_setting_2")
fig_scatter = plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=data, 
    x='setting1', 
    y='setting2', 
    hue='setting3', 
    style='setting3', 
    palette='coolwarm',
    s=100,
    edgecolor='black',
    alpha=0.8
)
plt.title('Relation entre op_setting_1 et op_setting_2', fontweight='bold', fontsize=16, color='darkgreen')
plt.xlabel('op_setting_1', fontweight='bold', fontsize=14, color='navy')
plt.ylabel('op_setting_2', fontweight='bold', fontsize=14, color='navy')
plt.legend(title='Setting 3', fontsize=12, title_fontsize=14)
plt.grid(visible=True, linestyle='--', alpha=0.6)
st.pyplot(fig_scatter)

# Distribution des RUL
st.subheader("Distribution de la durée de vie restante (RUL)")
fig_rul = plt.figure(figsize=(10, 5))
sns.histplot(data['RUL'], kde=True, bins=20, color='blue')
plt.title("Distribution du RUL")
st.pyplot(fig_rul)
# Fonction pour afficher une matrice de corrélation sous forme de carte thermique
def corr_matrix(data):
    fig, ax = plt.subplots(figsize=(12, 8))  # Taille ajustée pour une meilleure lisibilité
    
    # Contexte ajusté pour un rendu clair des polices
    sns.set_context("notebook", font_scale=1.2)  # Taille des polices augmentée
    
    # Carte thermique de la matrice de corrélation
    sns.heatmap(
        data.corr(), 
        annot=True,                # Affiche les valeurs numériques sur la carte
        fmt=".2f",                 # Format des nombres affichés
        cmap='coolwarm',           # Palette personnalisée pour une meilleure lecture
        linewidths=0.5,            # Ajout de lignes fines entre les cellules
        linecolor='gray',          # Couleur des lignes pour délimiter les cellules
        cbar_kws={'shrink': 0.8},  # Réduction de la taille de la barre de couleur
        ax=ax
    )
    
    # Titre de la matrice de corrélation
    ax.set_title('Matrice de corrélation', fontweight='bold', fontsize=16, color='darkblue')
    
    # Ajustement des marges pour éviter que les éléments soient coupés
    plt.tight_layout()

    return fig  # Retourne l'objet fig pour l'utiliser avec st.pyplot()

# Appelle la fonction pour afficher la matrice de corrélation
fig = corr_matrix(data)
st.pyplot(fig)  # Affiche la figure générée


def clean_data(train_data, test_data):
    # Liste des colonnes à supprimer
    cols_to_drop = ['setting1', 'setting2', 'sensor6', 'sensor5', 'sensor16', 'setting3', 'sensor1', 'sensor10', 'sensor18', 'sensor19']
    
    # Filtrer les colonnes existantes dans le DataFrame
    existing_cols_to_drop = [col for col in cols_to_drop if col in train_data.columns]
    
    # Supprimer les colonnes existantes dans le DataFrame
    cleaned_train = train_data.drop(existing_cols_to_drop, axis=1)
    cleaned_test = test_data.drop(existing_cols_to_drop, axis=1)
    
    return cleaned_train, cleaned_test

# Exemple d'appel de la fonction avec tes données
data = pd.read_csv('train_FD001.txt', sep=' ')  # Assure-toi que tes données sont correctement chargées
clean_train_data, clean_test_data = clean_data(data, data)  # Appliquer le nettoyage ici

# Afficher les résultats nettoyés pour vérifier
print(clean_train_data.head())
# Dictionnaire de noms de capteurs pour la visualisation
sens_names = {
    'sensor2': '(Température de sortie LPC) (◦R)',
    'sensor3': '(Température de sortie HPC) (◦R)',
    'sensor4': '(Température de sortie LPT) (◦R)',
    'sensor7': '(Pression de sortie HPC) (psia)',
    'sensor8': '(Vitesse physique du ventilateur) (rpm)',
    'sensor9': '(Vitesse physique du cœur) (rpm)',
    'sensor11': '(Pression statique de sortie HPC) (psia)',
    'sensor12': '(Rapport de débit de carburant à Ps30) (pps/psia)',
    'sensor13': '(Vitesse corrigée du ventilateur) (rpm)',
    'sensor14': '(Vitesse corrigée du cœur) (rpm)',
    'sensor15': '(Rapport de dérivation) ',
    'sensor17': '(Enthalpie de saignée)',
    'sensor20': '(Débit d’air de refroidissement des turbines à haute pression)',
    'sensor21': '(Débit d’air de refroidissement des turbines à basse pression)'
}

# Fonction pour afficher les courbes de chaque capteur
def plot_sensor(sensor_name, sens_names, data):
    for S in sensor_name:
        if S in data.columns:
            plt.figure(figsize=(13, 5))
            for i in data['engine'].unique():
                if (i % 5 == 0):
                    plt.plot('RUL', S, data=data[data['engine'] == i].rolling(8).mean())
            plt.xlim(250, 0)
            plt.xticks(np.arange(0, 275, 25))
            plt.ylabel(sens_names[S])
            plt.xlabel('Durée de vie utile restante (RUL)')
            plt.grid(True)
            plt.show()

# Charger les données
columns = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
@st.cache
def load_data():
    # Remplace par le chemin d'accès à ton fichier
    data = pd.read_csv("train_FD001.txt", delim_whitespace=True, header=None, names=columns)
    return data

data = load_data()

# Calculer la RUL
data['RUL'] = data.groupby('unit')['time'].transform(max) - data['time']

# Créer le label binaire
threshold = 50
data['label'] = (data['RUL'] <= threshold).astype(int)

# Interface utilisateur Streamlit
st.title('Analyse des moteurs Turbofan')
st.markdown('### Visualisation de la durée de vie restante (RUL) des moteurs')

# Nettoyage des données
st.sidebar.header("Sélection des Capteurs")
sensor_col = st.sidebar.multiselect("Choisir les capteurs", list(sens_names.keys()), default=list(sens_names.keys()))
clean_train_data, clean_test_data = clean_data(data, data)  # On applique le nettoyage ici

# Affichage des graphiques des capteurs choisis
if len(sensor_col) > 0:
    st.write("### Courbes de capteurs sélectionnés")
    plot_sensor(sensor_col, sens_names, clean_train_data)

# Réduction de la dimensionnalité avec PCA
st.markdown('### Analyse PCA (Réduction de la dimensionnalité)')
scaler = StandardScaler()
sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
data_scaled = scaler.fit_transform(data[sensor_columns])

pca = PCA(n_components=5)
data_pca = pca.fit_transform(data_scaled)

for i in range(data_pca.shape[1]):
    data[f'PC{i+1}'] = data_pca[:, i]

# Diviser les données
X = data[[f'PC{i+1}' for i in range(5)]]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression logistique
st.markdown('### Entraînement du modèle de régression logistique')
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
st.text(classification_rep)

# Visualisation des deux premières composantes principales
st.markdown('### Visualisation des deux premières composantes principales')
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['label'], cmap='viridis', alpha=0.7, edgecolor='k')
plt.colorbar(scatter, label='Label (0 = Normal, 1 = Critique)')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2') 
plt.title('Visualisation PCA avec classes')
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(plt)

 


# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime as dt

# Configuration de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Définition des colonnes du dataset
eng_cycle_col = ['engine', 'cycle']
setting_col = ['setting1', 'setting2', 'setting3']
sensor_col = [f'sensor{i}' for i in range(1, 22)]
columns = eng_cycle_col + setting_col + sensor_col

# Importation des données
train_data = pd.read_csv("train_FD001.txt", sep='\\s+', names=columns)
test_data = pd.read_csv("test_FD001.txt", sep='\\s+', names=columns)
true_rul = pd.read_csv("RUL_FD001.txt", sep='\\s+', names=['RUL'])

# Fonction pour ajouter la colonne "RUL"
def add_remaining_RUL(data):
    max_cycles = data.groupby('engine')['cycle'].max()
    merged = data.merge(max_cycles.to_frame(name='max_cycles'), left_on='engine', right_index=True)
    merged["RUL"] = merged["max_cycles"] - merged['cycle']
    return merged.drop("max_cycles", axis=1)

# Ajout de la colonne "RUL"
train_data = add_remaining_RUL(train_data)

# Suppression des colonnes inutiles pour l'entraînement
cols_to_drop = ['engine', 'cycle']
clean_train_data = train_data.drop(cols_to_drop, axis=1)

# Normalisation des données d'entraînement
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(clean_train_data.drop('RUL', axis=1))
scaled_data = pd.DataFrame(scaled_data, columns=clean_train_data.drop('RUL', axis=1).columns)

# Séparation des features (X) et de la cible (Y)
X_train = scaled_data
Y_train = train_data['RUL']

# Préparation des données de test
X_test = test_data.groupby('engine').last().reset_index()
X_test = X_test.drop(['setting1', 'setting2', 'sensor6', 'sensor5', 'sensor16',
                      'setting3', 'sensor1', 'sensor10', 'sensor18', 'sensor19'], axis=1)

scaled_test_data = scaler.transform(X_test.drop(['engine', 'cycle'], axis=1))
scaled_test_data = pd.DataFrame(scaled_test_data, columns=X_test.drop(['engine', 'cycle'], axis=1).columns)

Y_test = true_rul['RUL']
X_train_s = X_train
X_test_s = scaled_test_data

# Fonction d'évaluation du modèle
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE: {:.2f}, R2: {:.2f}'.format(label, rmse, variance))

# TEST 1: Régression Linéaire
start_1 = dt.now()
lm = LinearRegression()
lm.fit(X_train_s, Y_train)
Y_predict_train = lm.predict(X_train_s)
Y_predict_test = lm.predict(X_test_s)

print('Évaluation de la Régression Linéaire:')
print('Le temps d\'exécution est de : {}s'.format((dt.now() - start_1).seconds))

evaluate(Y_train, Y_predict_train, 'train')
evaluate(Y_test, Y_predict_test)

# Affichage de la courbe de la prédiction
plt.figure(figsize=(18, 10))
plt.plot(Y_test.values, color='red', label='RUL réel')
plt.plot(Y_predict_test, label='Prédiction Régression Linéaire')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# TEST 2: Arbre de Décision (Random Forest)
start_2 = dt.now()  # Enregistrement du temps de début pour l'arbre de décision
rf = RandomForestRegressor(max_features="sqrt", random_state=42)  # Initialisation du modèle Random Forest
rf.fit(X_train_s, Y_train)  # Entraînement du modèle avec les données d'entraînement
Y_predict_train_rf = rf.predict(X_train_s)  # Prédictions sur les données d'entraînement
Y_predict_test_rf = rf.predict(X_test_s)  # Prédictions sur les données de test

# Affichage du score du modèle Random Forest et du temps d'exécution
print('Évaluation du modèle Random Forest Regressor: ')
print('Le temps d\'exécution est de : ' + str((dt.now() - start_2).seconds) + 's')

# Évaluation du modèle Random Forest en utilisant la fonction `evaluate`
evaluate(Y_train, Y_predict_train_rf, 'train')  # Évaluation des prédictions sur les données d'entraînement
evaluate(Y_test, Y_predict_test_rf)  # Évaluation des prédictions sur les données de test

# Affichage de la courbe de la prédiction de l'arbre de décision
plt.plot(Y_predict_test_rf, color='orange', label='Prédiction Random Forest')  # Courbe de la prédiction
plt.legend(loc='upper left')
plt.grid(True)

# TEST 3: Machine à Vecteurs de Support (SVM)
start_3 = dt.now()  # Enregistrement du temps de début pour le modèle SVM
svm = SVR(kernel='linear')  # Initialisation du modèle SVM avec noyau linéaire
svm.fit(X_train_s, Y_train)  # Entraînement du modèle avec les données d'entraînement
svm_train_prediction = svm.predict(X_train_s)  # Prédictions sur les données d'entraînement
svm_test_predict = svm.predict(X_test_s)  # Prédictions sur les données de test

# Affichage du score du modèle SVM et du temps d'exécution
print('Évaluation du modèle Support Vector Machine: ')
print('Le temps d\'exécution est de : ' + str((dt.now() - start_3).seconds) + 's')

# Évaluation du modèle SVM en utilisant la fonction `evaluate`
evaluate(Y_train, svm_train_prediction, 'train')  # Évaluation des prédictions sur les données d'entraînement
evaluate(Y_test, svm_test_predict)  # Évaluation des prédictions sur les données de test

# Affichage de la courbe de la prédiction SVM
plt.plot(svm_test_predict, color='black', label='Prédiction SVM')  # Courbe de la prédiction SVM
plt.legend(loc='upper left')
plt.grid(True)

# Affichage de toutes les courbes
plt.show()
st.pyplot(fig)



