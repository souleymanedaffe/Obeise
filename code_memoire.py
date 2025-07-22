# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Configuration Streamlit ---
st.set_page_config(page_title="Détecteur d'Obésité", layout="centered")

# --- Chargement & entraînement du modèle (mis en cache) ---
@st.cache_resource
def load_and_train():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    # Discrétisation
    cuts_map = {
        'FCVC': [0,1.5,2.5,np.inf],
        'NCP':  [0,1.5,2.5,np.inf],
        'CH2O': [0,1.5,2.5,np.inf],
        'FAF':  [0,0.5,1.5,2.5,np.inf],
        'TUE':  [0,0.5,1.5,2.5,np.inf]
    }
    for col, bins in cuts_map.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

    # Calcul de l'IMC
    df['IMC'] = df['Weight'] / (df['Height']**2)

    # Encodage LabelEncoder
    le_dict = {}
    for col in ['Gender','family_history_with_overweight','FAVC','CAEC',
                'SMOKE','SCC','CALC','MTRANS','NObeyesdad']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Features & target
    features = [
        'IMC','Age','Height','Weight',
        'family_history_with_overweight','FAVC','FCVC','NCP','CAEC',
        'SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS'
    ]
    X = df[features]
    y = df['NObeyesdad']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )

    # Régression logistique
    model = LogisticRegression(
        penalty=None, solver='newton-cg', max_iter=1000, random_state=0
    )
    model.fit(X_train, y_train)

    return model, le_dict, features

model, le_dict, feature_columns = load_and_train()


# --- Connexion à Google Sheets via st.secrets ---
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
# st.secrets["gcp_service_account"] contient votre section TOML
sa_info = st.secrets["gcp_service_account"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("Réponses Obésité").sheet1


# --- Interface utilisateur ---
st.title("🔍 Détecteur de classe d'obésité")
st.write("Remplissez tous les champs, puis cliquez sur **Détecter l'obésité**.")

with st.form("input_form"):
    gender = st.selectbox("Genre", le_dict['Gender'].classes_)
    age    = st.number_input("Âge (années)", min_value=1, max_value=120, value=30)
    height = st.number_input("Taille (m)", step=0.01, value=1.70)
    weight = st.number_input("Poids (kg)", step=0.1, value=70.0)

    fh     = st.selectbox("Antécédent familial de surpoids",
                          le_dict['family_history_with_overweight'].classes_)
    favc   = st.selectbox("FAVC (aliments caloriques fréquents)",
                          le_dict['FAVC'].classes_)
    fcvc   = st.slider("FCVC (fréquence de légumes/jour)", 0.0, 7.0, 0.1, 2.0)
    ncp    = st.slider("NCP (nombre de repas principaux/jour)", 0.0, 5.0, 0.1, 2.0)
    caec   = st.selectbox("CAEC (grignotage entre les repas)",
                          le_dict['CAEC'].classes_)
    smoke  = st.selectbox("Fume", le_dict['SMOKE'].classes_)
    ch2o   = st.slider("CH2O (eau bue en L/jour)", 0.0, 5.0, 0.1, 2.0)
    scc    = st.selectbox("SCC (suivi des calories)", le_dict['SCC'].classes_)
    faf    = st.slider("FAF (heures d’activité physique/semaine)", 0.0, 20.0, 0.1, 1.0)
    tue    = st.slider("TUE (heures sur écrans/jour)", 0.0, 20.0, 0.1, 2.0)
    calc   = st.selectbox("CALC (consommation d’alcool)", le_dict['CALC'].classes_)
    mtrans = st.selectbox("MTRANS (transport principal)", le_dict['MTRANS'].classes_)

    submitted = st.form_submit_button("Détecter l'obésité")

if submitted:
    # Création du DataFrame d’entrée
    raw = pd.DataFrame([{
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history_with_overweight': fh, 'FAVC': favc,
        'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
        'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
        'CALC': calc, 'MTRANS': mtrans
    }])

    # Discrétisation identique
    for col, bins in [
        ('FCVC',[0,1.5,2.5,np.inf]),
        ('NCP',[0,1.5,2.5,np.inf]),
        ('CH2O',[0,1.5,2.5,np.inf]),
        ('FAF',[0,0.5,1.5,2.5,np.inf]),
        ('TUE',[0,0.5,1.5,2.5,np.inf])
    ]:
        raw[col] = pd.cut(raw[col], bins=bins, labels=False, include_lowest=True)

    # IMC
    raw['IMC'] = raw['Weight'] / (raw['Height']**2)

    # Encodage
    for col, le in le_dict.items():
        if col in raw.columns:
            raw[col] = le.transform(raw[col])

    # Prédiction
    X_new = raw[feature_columns]
    idx = model.predict(X_new)[0]
    class_map = [
        "Insuffisance pondérale",
        "Poids normal",
        "Surpoids (niveau I)",
        "Surpoids (niveau II)",
        "Obésité (type I)",
        "Obésité (type II)",
        "Obésité (type III)"
    ]
    pred_label = class_map[idx]
    st.success(f"**Classe prédite : {pred_label}**")

    # Préparation & envoi à Google Sheets
    raw['Classe prédite'] = pred_label
    row = raw.iloc[0].tolist()
    # Si première insertion, ajouter l’en‑tête
    if sheet.row_count == 0:
        header = raw.columns.tolist()
        sheet.append_row(header)
    sheet.append_row(row)

    st.info("Les données ont été enregistrées")

