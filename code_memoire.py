import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Configuration de la page
st.set_page_config(
    page_title="Détecteur d'Obésité",
    page_icon="💪",
    layout="centered",
    initial_sidebar_state="auto"
)

# 🎨 CSS personnalisé
st.markdown("""
    <style>
    body {
        background-color: #F0F2F6;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-size: 40px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
st.title(" Détecteur de classe d'obésité")

st.markdown("""
> Remplissez toutes les informations ci‑dessous puis cliquez sur **Détecter l'obésité**.
""")

# Fonction de chargement et entraînement
@st.cache_resource
def load_and_train():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    # Traduction des valeurs
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

    # Discrétisation
    cuts_map = {
        'FCVC': [0, 1.5, 2.5, np.inf],
        'NCP':  [0, 1.5, 2.5, np.inf],
        'CH2O': [0, 1.5, 2.5, np.inf],
        'FAF':  [0, 0.5, 1.5, 2.5, np.inf],
        'TUE':  [0, 0.5, 1.5, 2.5, np.inf]
    }
    for col, bins in cuts_map.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

    # IMC
    df['IMC'] = df['Weight'] / (df['Height']**2)

    # Encodage
    le_dict = {}
    for col in ['Gender','family_history_with_overweight','FAVC','CAEC',
                'SMOKE','SCC','CALC','MTRANS','NObeyesdad']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    features = [
        'IMC','Age','Height','Weight',
        'family_history_with_overweight','FAVC','FCVC','NCP','CAEC',
        'SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS'
    ]
    X = df[features]
    y = df['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )

    model = LogisticRegression(
        penalty=None, solver='newton-cg', max_iter=1000, random_state=0
    )
    model.fit(X_train, y_train)

    return model, le_dict, features

model, le_dict, feature_columns = load_and_train()

# Formulaire
with st.form("input_form"):
    gender = st.selectbox("Genre", le_dict['Gender'].classes_)
    age    = st.number_input("Âge (années)", min_value=1, max_value=120, value=20)
    height = st.number_input("Taille (m)", step=0.01, value=1.70)
    weight = st.number_input("Poids (kg)", step=0.1,  value=70.0)
    fh     = st.selectbox("Antécédent familial de surpoids", le_dict['family_history_with_overweight'].classes_)
    favc   = st.selectbox("FAVC (aliments caloriques fréquents)", le_dict['FAVC'].classes_)
    fcvc   = st.slider("FCVC (fréquence de légumes/jour)", 0.0, 7.0, step=0.1, value=2.0)
    ncp    = st.slider("NCP (repas principaux/jour)", 0.0, 5.0, step=0.1, value=2.0)
    caec   = st.selectbox("CAEC (grignotage entre repas)", le_dict['CAEC'].classes_)
    smoke  = st.selectbox("Fume", le_dict['SMOKE'].classes_)
    ch2o   = st.slider("CH2O (litres d'eau / jour)", 0.0, 5.0, step=0.1, value=2.0)
    scc    = st.selectbox("SCC (suivi des calories)", le_dict['SCC'].classes_)
    faf    = st.slider("FAF (activité physique / semaine)", 0.0, 20.0, step=0.1, value=1.0)
    tue    = st.slider("TUE (temps écran / jour)", 0.0, 20.0, step=0.1, value=2.0)
    calc   = st.selectbox("CALC (alcool)", le_dict['CALC'].classes_)
    mtrans = st.selectbox("Moyen de transport", le_dict['MTRANS'].classes_)

    submitted = st.form_submit_button("🚀 Détecter l'obésité")

# Traitement après soumission
DATA_FILE = "user_inputs.xlsx"

if submitted:
    raw = pd.DataFrame([{
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history_with_overweight': fh, 'FAVC': favc,
        'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
        'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
        'CALC': calc, 'MTRANS': mtrans
    }])

    for col, bins in [
        ('FCVC', [0,1.5,2.5,np.inf]),
        ('NCP',  [0,1.5,2.5,np.inf]),
        ('CH2O', [0,1.5,2.5,np.inf]),
        ('FAF',  [0,0.5,1.5,2.5,np.inf]),
        ('TUE',  [0,0.5,1.5,2.5,np.inf])
    ]:
        raw[col] = pd.cut(raw[col], bins=bins, labels=False, include_lowest=True)

    raw['IMC'] = raw['Weight'] / (raw['Height']**2)

    for col, le in le_dict.items():
        if col in raw.columns:
            raw[col] = le.transform(raw[col])

    X_new = raw[feature_columns]
    pred_index = model.predict(X_new)[0]
    pred_label = le_dict['NObeyesdad'].inverse_transform([pred_index])[0]

    st.success(f" Classe prédite : **{pred_label}**")

    raw['Predicted_Class'] = pred_label

    if os.path.exists(DATA_FILE):
        df_hist = pd.read_excel(DATA_FILE, engine='openpyxl')
        df_all = pd.concat([df_hist, raw], ignore_index=True)
    else:
        df_all = raw.copy()

    df_all.to_excel(DATA_FILE, index=False, engine='openpyxl')
    st.info(f" Les données ont été enregistrées avec succès.")

# 🖋️ Signature en bas de page
st.markdown("""
<div class="footer">
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST</strong> 
</div>
""", unsafe_allow_html=True)
