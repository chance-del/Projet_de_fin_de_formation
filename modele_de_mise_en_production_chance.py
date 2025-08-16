import streamlit as st
import pandas as pd
import joblib
import openpyxl

# ==========================================
# 0. Config générale
# ==========================================
st.set_page_config(
    page_title="Prédiction Éligibilité Crédit",
    page_icon="💳",
    layout="wide"
)


# --- Définition des couleurs et police ---
primary_color = "#D28E8E"
background_color = "#528D4E"
secondary_background_color = "#F0F2F6"
text_color = "#31333F"
font_family = "sans-serif" # Correspond à "Sans empattement"

# --- Injection de CSS personnalisé ---
st.markdown(
    f"""
    <style>
    /* Général (corps de la page) */
    body {{
        color: {text_color};
        background-color: {background_color};
        font-family: {font_family};
    }}

    /* Styles pour le conteneur principal de Streamlit */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: {font_family};
    }}

    /* Barre latérale et éléments similaires (secondary_background_color) */
    .stSidebar {{
        background-color: {secondary_background_color};
        color: {text_color}; /* Le texte dans la sidebar devrait être lisible */
    }}

    .st-emotion-cache-16txt3u {{ /* Cible le fond des conteneurs par exemple */
        background-color: {secondary_background_color};
    }}

    .st-emotion-cache-zq5wmm {{ /* Cible le fond des conteneurs par exemple */
        background-color: {secondary_background_color};
    }}

    /* Boutons (primary_color) */
    .stButton>button {{
        background-color: {primary_color};
        color: white; /* Texte blanc sur bouton primaire pour un bon contraste */
        border: none;
    }}

    /* Curseurs (sliders) - la couleur principale est souvent utilisée ici */
    .stSlider>div>div>div>div {{ /* La barre du curseur */
        background-color: {primary_color};
    }}
    .stSlider>div>div>div>div>div[data-testid="stSliderHandle"] {{ /* Le "pouce" du curseur */
        background-color: {primary_color};
    }}

    /* Entrées de texte, zones de texte (peut aussi utiliser primaryColor ou textColor) */
    .stTextInput>div>div>input {{
        color: {text_color};
        background-color: {secondary_background_color};
        border-color: {primary_color}; /* Bordure pour les inputs */
    }}
    .stTextArea>div>div>textarea {{
        color: {text_color};
        background-color: {secondary_background_color};
        border-color: {primary_color};
    }}

    /* Titres (H1, H2, etc.) */
    h1, h2, h3, h4, h5, h6 {{
        color: {primary_color}; /* Utiliser la couleur primaire pour les titres peut être sympa */
        font-family: {font_family};
    }}

    /* Texte général */
    p, li, div, span {{
        color: {text_color};
        font-family: {font_family};
    }}

    /* Liens */
    a {{
        color: {primary_color};
    }}

    /* Expander / Checkbox / Radio / Selectbox */
    .st-emotion-cache-1f1c24p, /* Checkbox label */
    .st-emotion-cache-1cpxd0t, /* Radio label */
    .st-emotion-cache-1oe5f0g, /* Selectbox label */
    .st-emotion-cache-1c09d5y, /* Expander header */
    .st-emotion-cache-1y4y1h6 {{ /* Expander header content */
        color: {text_color};
    }}

    .st-emotion-cache-v0n0as, /* Background of selectbox options */
    .st-emotion-cache-j9f0gy,
    .st-emotion-cache-1c9s62z {{ /* Hover background of selectbox options */
        background-color: {secondary_background_color} !important;
        color: {text_color} !important;
    }}

    /* La couleur de survol pour les options sélectionnées */
    .st-emotion-cache-1c9s62z:hover {{
        background-color: {primary_color} !important;
        color: white !important;
    }}

    /* Éléments st.container avec bordure */
    .stContainer {{
        background-color: {secondary_background_color};
        border: 1px solid {primary_color}; /* Ajoute une bordure pour distinguer */
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: space-around;
    }
    </style>
    """, unsafe_allow_html=True)
# ==========================================
# 1. Charger le modèle MLP sauvegardé
# ==========================================
import joblib
import zipfile
import os
import tempfile
import streamlit as st

@st.cache_resource
def load_model():
    zip_path = "mlp_pipeline_model.zip"  # ton fichier zippé (placé dans ton repo GitHub)
    
    # Décompresser dans un dossier temporaire
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_dir = tempfile.mkdtemp()
        zip_ref.extractall(temp_dir)

    # Chercher le .pkl dans le zip
    model_path = None
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".pkl"):
                model_path = os.path.join(root, file)
                break
    
    if model_path is None:
        raise FileNotFoundError("Aucun fichier .pkl trouvé dans le ZIP.")
    
    # Charger le modèle avec joblib
    model = joblib.load(model_path)
    return model


pipeline = load_model()

# ==========================================
# 2. Liste des features utilisées
# ==========================================
features = [
    'service_slass_encoded', 'genre_encoded', 'region_adamaoua',
    'region_centre', 'region_east', 'region_far north', 'region_littoral',
    'region_north', 'region_north west', 'region_south',
    'region_south west', 'region_west', 'nb_day_activity_avg',
    'nb_mois_anciennete', 'incoming_onnet_cmt_sms_avg', 'rev_voice_avg',
    'incoming_onnet_cmt_calls_avg', 'incoming_int_duration_avg',
    'valeur_remit_in_avg', 'incoming_onnet_ocm_calls_avg',
    'valeur_cashout_avg', 'volume_remit_in_avg', 'valeur_p2p_avg',
    'volume_cashin_avg', 'incoming_onnet_nb_sms_avg', 'volume_payment_avg',
    'valeur_remit_out_avg', 'incoming_onnet_ocm_sms_avg', 'xtratime_avg',
    'aspu_vdsd_avg', 'incoming_onnet_int_sms_avg', 'rev_sms_avg',
    'incoming_onnet_duration_avg', 'ma_aspu_avg', 'volume_p2p_avg',
    'incoming_cmt_duration_avg', 'valeur_payment_avg',
    'incoming_ocm_duration_avg', 'incoming_onnet_nb_calls_avg',
    'valeur_cashin_avg', 'incoming_onnet_int_calls_avg', 'age'
]

# ==========================================
# 3. Navigation multipage
# ==========================================
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Aller à :", ["Prédiction client unique", "Prédiction multiple"])

# ==========================================
# Page 1 : prédiction pour un seul client
# ==========================================
if page == "Prédiction client unique":
    st.title("🔹 Prédiction d'éligibilité (Client unique)")
    st.write("Saisissez les informations du client :")

    input_data = {}

    with st.form(key="user_input_form"):
        # Variables catégorielles encodées
        input_data['genre_encoded'] = st.selectbox(
            "Genre", [0,1], format_func=lambda x: "F" if x==0 else "M"
        )
        input_data['service_slass_encoded'] = st.selectbox(
            "Type de service", [0,1], format_func=lambda x: "prepaid" if x==0 else "pospaid"
        )

        # Région unique
        region_cols = [f for f in features if f.startswith("region_")]
        region_choice = st.selectbox(
            "Sélectionnez la région :", 
            [r.replace("region_", "").capitalize() for r in region_cols]
        )
        for r in region_cols:
            input_data[r] = 0
        chosen_col = "region_" + region_choice.lower()
        if chosen_col in input_data:
            input_data[chosen_col] = 1

        # Variables numériques (5 colonnes par ligne)
        numeric_cols = [f for f in features if f not in ['genre_encoded','service_slass_encoded'] + region_cols]
        n = 5
        for i in range(0, len(numeric_cols), n):
            cols = st.columns(n)
            for j, col in enumerate(numeric_cols[i:i+n]):
                with cols[j]:
                    input_data[col] = st.number_input(col, value=0.0)

        submit_button = st.form_submit_button("Prédire")

    if submit_button:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[features]  # ordre correct
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df).max()

        result_text = "✅ Éligible" if prediction == 1 else "❌ Non éligible"

        st.subheader("Résultat :")
        st.write(f"**Classe prédite :** {result_text}")
        st.write(f"**Confiance du modèle :** {prediction_proba:.2%}")

# ==========================================
# Page 2 : prédiction pour plusieurs clients
# ==========================================
elif page == "Prédiction multiple":
    st.title("🔹 Prédiction d'éligibilité (Plusieurs clients)")
    st.write("Chargez un fichier CSV/Excel contenant les caractéristiques des clients.")

    uploaded_file = st.file_uploader("Uploader un fichier", type=["csv","xlsx"])

    if uploaded_file is not None:
        # Charger les données
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("Aperçu des données chargées :")
        st.dataframe(data.head())

        # Vérifier que les colonnes attendues sont présentes
        missing_cols = [col for col in features if col not in data.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes dans le fichier : {missing_cols}")
        else:
            # Prédiction
            predictions = pipeline.predict(data[features])
            probas = pipeline.predict_proba(data[features]).max(axis=1)

            results = data.copy()
            results["prediction"] = ["✅ Éligible" if p==1 else "❌ Non éligible" for p in predictions]
            results["confiance"] = probas

            st.write("Résultats des prédictions :")
            st.dataframe(results.head(20))

            # Bouton de téléchargement
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger les prédictions",
                data=csv,
                file_name="predictions_clients.csv",
                mime="text/csv"
            )

