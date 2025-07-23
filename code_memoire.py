# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Détecteur d'Obésité", layout="centered")

@st.cache_resource
def load_and_train():
    # 1) Chargement du dataset
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    # 2) Traduction des valeurs en français
    df['Gender'].replace({'Male': 'Homme', 'Female': 'Femme'}, inplace=True)
    df['family_history_with_overweight'].replace({'yes': 'Oui', 'no': 'Non'}, inplace=True)
    df['FAVC'].replace({'yes': 'Oui', 'no': 'Non'}, inplace=True)
    df['CAEC'].replace({'no': 'Non', 'Sometimes': 'Parfois', 'Frequently': 'Fréquemment', 'Always': 'Toujours'}, inplace=True)
    df['SMOKE'].replace({'yes': 'Oui', 'no': 'Non'}, inplace=True)
    df['SCC'].replace({'yes': 'Oui', 'no': 'Non'}, inplace=True)
    df['CALC'].replace({'no': 'Non', 'Sometimes': 'Parfois', 'Frequently': 'Fréquemment', 'Always': 'Toujours'}, inplace=True)
    df['MTRANS'].replace({
        'Public_Transportation': 'Transport public',
        'Walking': 'Marche',
        'Automobile': 'Voiture',
        'Motorbike': 'Moto',
        'Bike': 'Vélo'
    }, inplace=True)
    df['NObeyesdad'].replace({
        'Insufficient_Weight': 'Insuffisance pondérale',
        'Normal_Weight': 'Poids normal',
        'Overweight_Level_I': 'Surpoids (niveau I)',
        'Overweight_Level_II': 'Surpoids (niveau II)',
        'Obesity_Type_I': 'Obésité (type I)',
        'Obesity_Type_II': 'Obésité (type II)',
        'Obesity_Type_III': 'Obésité (type III)'
    }, inplace=True)

    # 3) Discrétisation des variables continues
    cuts_map = {
        'FCVC': [0, 1.5, 2.5, np.inf],
        'NCP':  [0, 1.5, 2.5, np.inf],
        'CH2O': [0, 1.5, 2.5, np.inf],
        'FAF':  [0, 0.5, 1.5, 2.5, np.inf],
        'TUE':  [0, 0.5, 1.5, 2.5, np.inf]
    }
    for col, bins in cuts_map.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

    # 4) Calcul de l'IMC
    df['IMC'] = df['Weight'] / (df['Height']**2)

    # 5) Encodage des variables catégorielles
    le_dict = {}
    for col in ['Gender','family_history_with_overweight','FAVC','CAEC',
                'SMOKE','SCC','CALC','MTRANS','NObeyesdad']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # 6) Sélection des caractéristiques et de la cible
    features = [
        'IMC','Age','Height','Weight',
        'family_history_with_overweight','FAVC','FCVC','NCP','CAEC',
        'SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS'
    ]
    X = df[features]
    y = df['NObeyesdad']

    # 7) Partition train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )

    # 8) Entraînement du modèle
    model = LogisticRegression(
        penalty=None, solver='newton-cg', max_iter=1000, random_state=0
    )
    model.fit(X_train, y_train)

    return model, le_dict, features

model, le_dict, feature_columns = load_and_train()

# --- Interface utilisateur ---
st.title("Détecteur de classe d'obésité")
st.write("Remplissez toutes les informations ci‑dessous puis cliquez sur **Détecter l'obésité**.")

with st.form("input_form"):
    gender = st.selectbox("Genre", le_dict['Gender'].classes_)
    age    = st.number_input("Âge (années)", min_value=1, max_value=120, value=20)
    height = st.number_input("Taille (m)", step=0.01, value=1.70)
    weight = st.number_input("Poids (kg)", step=0.1,  value=70.0)

    fh     = st.selectbox("Antécédent familial de surpoids", 
                          le_dict['family_history_with_overweight'].classes_)
    favc   = st.selectbox("FAVC (aliments caloriques fréquents)", 
                          le_dict['FAVC'].classes_)
    fcvc   = st.slider("FCVC (fréquence de légumes/jour)", 0.0, 7.0, step=0.1, value=2.0)
    ncp    = st.slider("NCP (nombre de repas principaux/jour)", 0.0, 5.0, step=0.1, value=2.0)
    caec   = st.selectbox("CAEC (grignotage entre les repas)", 
                          le_dict['CAEC'].classes_)
    smoke  = st.selectbox("Fume", le_dict['SMOKE'].classes_)
    ch2o   = st.slider("CH2O (eau bue en litres/jour)", 0.0, 5.0, step=0.1, value=2.0)
    scc    = st.selectbox("SCC (suivi des calories)", le_dict['SCC'].classes_)
    faf    = st.slider("FAF (heures d’activité physique/semaine)", 0.0, 20.0, step=0.1, value=1.0)
    tue    = st.slider("TUE (heures passées sur écrans/jour)", 0.0, 20.0, step=0.1, value=2.0)
    calc   = st.selectbox("CALC (consommation d’alcool)", le_dict['CALC'].classes_)
    mtrans = st.selectbox("MTRANS (moyen de transport principal)", 
                          le_dict['MTRANS'].classes_)

    submitted = st.form_submit_button("Détecter l'obésité")

# Chemin du fichier Excel où vous stockez les réponses
DATA_FILE = "user_inputs.xlsx"

if submitted:
    # Création d’un DataFrame pour l’individu saisi
    raw = pd.DataFrame([{
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history_with_overweight': fh, 'FAVC': favc,
        'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
        'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
        'CALC': calc, 'MTRANS': mtrans
    }])

    # Discrétisation identique à l’entraînement
    for col, bins in [
        ('FCVC', [0,1.5,2.5,np.inf]),
        ('NCP',  [0,1.5,2.5,np.inf]),
        ('CH2O', [0,1.5,2.5,np.inf]),
        ('FAF',  [0,0.5,1.5,2.5,np.inf]),
        ('TUE',  [0,0.5,1.5,2.5,np.inf])
    ]:
        raw[col] = pd.cut(raw[col], bins=bins, labels=False, include_lowest=True)

    # Calcul de l'IMC
    raw['IMC'] = raw['Weight'] / (raw['Height']**2)

    # Encodage des catégories
    for col, le in le_dict.items():
        if col in raw.columns:
            raw[col] = le.transform(raw[col])

    # Prédiction
    X_new = raw[feature_columns]
    pred_index = model.predict(X_new)[0]
    pred_label = le_dict['NObeyesdad'].inverse_transform([pred_index])[0]

    st.success(f"**Classe prédite : {pred_label}**")

    # Ajout de la colonne prédite
    raw['Predicted_Class'] = pred_label

    # Sauvegarde
    if os.path.exists(DATA_FILE):
        df_hist = pd.read_excel(DATA_FILE, engine='openpyxl')
        df_all = pd.concat([df_hist, raw], ignore_index=True)
    else:
        df_all = raw.copy()

    df_all.to_excel(DATA_FILE, index=False, engine='openpyxl')
    st.info(f"Les données ont été enregistrées")
