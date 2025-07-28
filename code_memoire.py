import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Configuration de la page
st.set_page_config(
    page_title="Détecteur d'Obésité",
    page_icon="",
    layout="centered"
)

# CSS personnalisé
st.markdown("""
    <style>
    .stApp {
       
        font-size: 50px !important;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        font-size: 42px !important;
        color: #ff4b4b;
        text-align: center;
    }

    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 30px !important;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 50px !important;
        font-weight: bold;
        padding: 0.7em 1.2em;
        border-radius: 8px;
    }

    .footer {
        text-align: center;
        font-size: 40px !important;
        margin-top: 50px;
        color: gray;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
st.title(" Détecteur de classe d'obésité")
st.markdown("Remplissez les informations ci-dessous puis cliquez sur **Détecter l'obésité**.")

# Chargement et entraînement
@st.cache_resource
def load_and_train():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

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

    cuts_map = {
        'FCVC': [0, 1.5, 2.5, np.inf],
        'NCP':  [0, 1.5, 2.5, np.inf],
        'CH2O': [0, 1.5, 2.5, np.inf],
        'FAF':  [0, 0.5, 1.5, 2.5, np.inf],
        'TUE':  [0, 0.5, 1.5, 2.5, np.inf]
    }
    for col, bins in cuts_map.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

    df['IMC'] = df['Weight'] / (df['Height']**2)

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

# Formulaire utilisateur
with st.form("input_form"):
    gender = st.selectbox("Genre", le_dict['Gender'].classes_)
    age = st.number_input("Âge (années)", min_value=1, max_value=120, value=25)
    height = st.number_input("Taille (m)", step=0.01, value=1.70)
    weight = st.number_input("Poids (kg)", step=0.1, value=70.0)
    fh = st.selectbox("Antécédents familiaux de surpoids", le_dict['family_history_with_overweight'].classes_)
    favc = st.selectbox("Consommation fréquente d'aliments caloriques (FAVC)", le_dict['FAVC'].classes_)
    fcvc = st.slider("Fréquence de consommation de légumes (FCVC)", 0.0, 7.0, 2.0, step=0.1)
    ncp = st.slider("Nombre de repas principaux par jour (NCP)", 0.0, 5.0, 2.0, step=0.1)
    caec = st.selectbox("Grignotage entre les repas (CAEC)", le_dict['CAEC'].classes_)
    smoke = st.selectbox("Fumez-vous ?", le_dict['SMOKE'].classes_)
    ch2o = st.slider("Consommation d'eau par jour (litres)", 0.0, 5.0, 2.0, step=0.1)
    scc = st.selectbox("Surveillance des calories (SCC)", le_dict['SCC'].classes_)
    faf = st.slider("Activité physique (heures/semaine)", 0.0, 20.0, 1.0, step=0.1)
    tue = st.slider("Temps passé devant un écran (heures/jour)", 0.0, 20.0, 2.0, step=0.1)
    calc = st.selectbox("Consommation d'alcool", le_dict['CALC'].classes_)
    mtrans = st.selectbox("Moyen de transport principal", le_dict['MTRANS'].classes_)

    submitted = st.form_submit_button(" Détecter l'obésité")

# Prédiction et affichage des résultats
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
        ('FCVC', [0, 1.5, 2.5, np.inf]),
        ('NCP',  [0, 1.5, 2.5, np.inf]),
        ('CH2O', [0, 1.5, 2.5, np.inf]),
        ('FAF',  [0, 0.5, 1.5, 2.5, np.inf]),
        ('TUE',  [0, 0.5, 1.5, 2.5, np.inf])
    ]:
        raw[col] = pd.cut(raw[col], bins=bins, labels=False, include_lowest=True)

    raw['IMC'] = raw['Weight'] / (raw['Height']**2)

    for col, le in le_dict.items():
        if col in raw.columns:
            raw[col] = le.transform(raw[col])

    X_new = raw[feature_columns]
    pred_index = model.predict(X_new)[0]
    pred_label = le_dict['NObeyesdad'].inverse_transform([pred_index])[0]

    imc_value = float(raw['IMC'].iloc[0])
    st.markdown(f"###  IMC calculé : **{imc_value:.2f} kg/m²**")
    st.success(f" Classe prédite : **{pred_label}**")

    # Affichage des conseils
    if imc_value < 18.5:
        st.warning("🧍 IMC faible → *Insuffisance pondérale*. Consultez un nutritionniste.")
    elif 18.5 <= imc_value < 25.0:
        st.success("🏃 IMC normal → Continuez votre mode de vie équilibré.")
    elif 25.0 <= imc_value < 30.0:
        st.info("⚠️ Surpoids → Réduisez les aliments caloriques et bougez plus.")
    elif 30.0 <= imc_value < 35.0:
        st.warning("🚨 Obésité modérée → Accompagnement médical recommandé.")
    elif 35.0 <= imc_value < 40.0:
        st.error("❗ Obésité sévère → Suivi médical nécessaire.")
    else:
        st.error("⛔ Obésité massive → Intervention médicale urgente requise.")

    raw['Predicted_Class'] = pred_label

    if os.path.exists(DATA_FILE):
        df_hist = pd.read_excel(DATA_FILE, engine='openpyxl')
        df_all = pd.concat([df_hist, raw], ignore_index=True)
    else:
        df_all = raw.copy()

    df_all.to_excel(DATA_FILE, index=False, engine='openpyxl')
   

# Signature
st.markdown("""
<div class="footer">
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST</strong>
</div>
""", unsafe_allow_html=True)
