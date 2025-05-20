# -*- coding: utf-8 -*-
import os
import flask
import logging
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import gdown
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for testing; restrict in production
app.secret_key = 'your-secret-key-123'  # Replace with a secure key in production

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictor_app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and encoders
model = None
le_region = None
le_gender = None
le_prev_meds = None
le_disease = None
le_drug = None
le_tablets = None
scaler = None
feature_names = None

# File IDs for .pkl files on Google Drive (updated based on new list)
FILE_IDS = {
    "disease_model.pkl": "1A8QBshgq224QqO6yos7dhTQeqe8vXbSN",
    "feature_names.pkl": "1C4jhR3h_x9KmGTTrQm__VcrtFktdjZee",
    "le_disease.pkl": "1HlugauFkKy__JImJIO97s1tZpADxG9bm",
    "le_drug.pkl": "10cxbwDWjR3bMMErQWjArb3sReP-Xe3YP",
    "le_gender.pkl": "1ATyQLRPL-Nj-mbglLv9ABsopKQ4J_da0",
    "le_prev_meds.pkl": "1e9Bd8gnmrh0EkKm93Xt_kL6rISGwYzmv",
    "le_region.pkl": "1tL4f8THUNYb8XKx2PQKgxmDZDfGeipGC",
    "le_tablets.pkl": "1d4lwa744xPB2SFmC-L94_UK2R9OtDhUI",
    "scaler.pkl": "12mhenLacpObXFMShsxA2FrSJt0l-3qd8"  # Assumed to be malaria_model.pkl
}

def download_models():
    """Download .pkl files from Google Drive using gdown."""
    for file_name, file_id in FILE_IDS.items():
        if not os.path.exists(file_name):
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                logger.info(f"Downloading {file_name} from Google Drive...")
                gdown.download(url, file_name, quiet=False)
                logger.info(f"Downloaded {file_name} successfully")
            except Exception as e:
                logger.error(f"Failed to download {file_name}: {e}")
                raise

def load_models():
    """Load .pkl files into global variables."""
    global model, le_region, le_gender, le_prev_meds, le_disease, le_drug, le_tablets, scaler, feature_names
    try:
        model = joblib.load("disease_model.pkl")
        le_region = joblib.load("le_region.pkl")
        le_gender = joblib.load("le_gender.pkl")
        le_prev_meds = joblib.load("le_prev_meds.pkl")
        le_disease = joblib.load("le_disease.pkl")
        le_drug = joblib.load("le_drug.pkl")
        le_tablets = joblib.load("le_tablets.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
        logger.info("Model, encoders, scaler, and feature names loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

# Download and load models at startup
try:
    download_models()
    load_models()
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    model = None  # Ensure model remains None if initialization fails

@app.route('/')
def index():
    return flask.jsonify({
        'message': 'Welcome to the Disease Predictor API',
        'endpoints': {
            '/health': 'Check server status',
            '/predict': 'Submit symptoms for disease prediction'
        },
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model, le_region, le_gender, le_prev_meds, le_disease, le_drug, le_tablets, scaler, feature_names
    if model is None:
        logger.error("Model not loaded")
        return flask.jsonify({'error': 'Model not initialized. Check server logs for initialization errors.'}), 500

    try:
        global VALID_PREV_MEDS
        if not VALID_PREV_MEDS and le_prev_meds:
            VALID_PREV_MEDS = list(le_prev_meds.classes_)
        
        data = flask.request.get_json()
        logger.info(f"Received data: {data}")

        if not data:
            logger.warning("Empty request data")
            return flask.jsonify({"error": "No data provided"}), 400

        input_data = {feature: 0 for feature in feature_names}
        for key in VALID_SYMPTOMS + ['Pregnant', 'First_Trimester_Pregnant', 'G6PD_Deficiency']:
            input_data[key] = int(data.get(key, 0))

        try:
            input_data['Age'] = int(data.get('Age', 0))
            raw_weight = float(data.get('Weight', 0))
            input_data['Weight'] = raw_weight
        except ValueError:
            logger.warning("Invalid Age or Weight format")
            return flask.jsonify({"error": "Age and Weight must be numbers"}), 400

        input_data['Region'] = data.get('Region', '')
        input_data['Gender'] = data.get('Gender', '')
        input_data['Previous_Medications'] = data.get('Previous_Medications', '')

        if input_data['Age'] < 1 or input_data['Age'] > 120:
            logger.warning(f"Invalid age: {input_data['Age']}")
            return flask.jsonify({"error": "Age must be between 1 and 120"}), 400
        if raw_weight < 1 or raw_weight > 200:
            logger.warning(f"Invalid weight: {raw_weight}")
            return flask.jsonify({"error": "Weight must be between 1 and 200 kg"}), 400
        if input_data['Region'] not in VALID_REGIONS:
            logger.warning(f"Invalid region: {input_data['Region']}")
            return flask.jsonify({"error": f"Region must be one of {VALID_REGIONS}"}), 400
        if input_data['Gender'] not in VALID_GENDERS:
            logger.warning(f"Invalid gender: {input_data['Gender']}")
            return flask.jsonify({"error": "Gender must be Male or Female"}), 400
        if input_data['Previous_Medications'] not in VALID_PREV_MEDS:
            logger.warning(f"Invalid previous medications: {input_data['Previous_Medications']}")
            input_data['Previous_Medications'] = VALID_PREV_MEDS[0]
            logger.info(f"Defaulted Previous_Medications to {input_data['Previous_Medications']}")
        if input_data['Pregnant'] == 1 and input_data['Gender'] != 'Female':
            logger.warning("Pregnant selected for non-female gender")
            return flask.jsonify({"error": "Pregnant can only be selected for Female gender"}), 400
        if input_data['First_Trimester_Pregnant'] == 1 and input_data['Pregnant'] != 1:
            logger.warning("First Trimester Pregnant selected without Pregnant")
            return flask.jsonify({"error": "First Trimester Pregnant requires Pregnant to be Yes"}), 400

        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
            logger.warning(f"Missing features in input_data: {missing_features}")

        severe_malaria_symptoms = [
            'impaired_consciousness', 'prostration', 'convulsions', 'deep_breathing',
            'respiratory_distress', 'abnormal_bleeding', 'severe_anemia', 'severe_dehydration'
        ]
        if any(input_data.get(s, 0) == 1 for s in severe_malaria_symptoms):
            logger.warning("Severe malaria symptoms detected")
            return flask.jsonify({
                "error": "Severe malaria detected. Refer to a health facility for parenteral treatment (Quinine or Artemisinin derivatives)."
            }), 400

        severe_dengue_symptoms = ['impaired_consciousness', 'mucosal_bleeding', 'respiratory_distress']
        dengue_warning_signs = ['abdominal_pain', 'vomiting', 'lethargy']
        severe_dengue = any(input_data.get(s, 0) == 1 for s in severe_dengue_symptoms)
        warning_signs = any(input_data.get(s, 0) == 1 for s in dengue_warning_signs)
        logger.info(f"Severe dengue: {severe_dengue}, Warning signs: {warning_signs}")

        typhoid_complications = ['delirium']
        if any(input_data.get(s, 0) == 1 for s in typhoid_complications):
            logger.warning("Typhoid complications detected")
            return flask.jsonify({
                "error": "Typhoid complications detected. Refer to HC4 for IV antibiotics (Ceftriaxone)."
            }), 400

        symptom_keys = [k for k in input_data if k in VALID_SYMPTOMS]
        if not any(input_data[k] == 1 for k in symptom_keys):
            logger.warning("No symptoms selected")
            return flask.jsonify({"error": "At least one symptom must be selected"}), 400

        try:
            input_data['Region'] = le_region.transform([input_data['Region']])[0]
            input_data['Gender'] = le_gender.transform([input_data['Gender']])[0]
            input_data['Previous_Medications'] = le_prev_meds.transform([input_data['Previous_Medications']])[0]
        except ValueError as e:
            logger.error(f"Label encoding error: {e}")
            return flask.jsonify({"error": f"Invalid categorical value: {e}"}), 400

        numerical_cols = ['Age', 'Weight']
        numerical_data = pd.DataFrame([[input_data['Age'], raw_weight]], columns=numerical_cols)
        scaled_data = scaler.transform(numerical_data)
        input_data['Age'] = scaled_data[0, 0]
        input_data['Weight'] = scaled_data[0, 1]

        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_df)[0]

        disease = le_disease.inverse_transform([prediction[0]])[0]
        predicted_drug = le_drug.inverse_transform([prediction[1]])[0]
        predicted_tablets = le_tablets.inverse_transform([prediction[2]])[0]

        if disease.lower().find('malaria') != -1:
            treatment = get_dosage(
                weight=raw_weight,
                drug=predicted_drug,
                pregnant=bool(input_data['Pregnant']),
                first_trimester=bool(input_data['First_Trimester_Pregnant']),
                g6pd_deficiency=bool(input_data['G6PD_Deficiency'])
            )
        elif disease.lower().find('dengue') != -1:
            treatment = get_dengue_dosage(
                weight=raw_weight,
                warning_signs=warning_signs,
                severe=severe_dengue
            )
        elif disease.lower().find('typhoid') != -1:
            treatment = get_typhoid_dosage(
                weight=raw_weight,
                resistance=False
            )
        else:
            logger.info(f"Non-malaria, non-dengue, non-typhoid disease predicted: {disease}")
            treatment = {
                "drug": predicted_drug,
                "dosage": {"tablets": int(predicted_tablets), "note": "Dosage based on model prediction; consult a health worker."},
                "warnings": ["Unknown disease; treatment may not align with guidelines."],
                "supportive": get_paracetamol_dosage(raw_weight) if input_data['fever'] else None
            }

        if 'error' in treatment:
            logger.error(f"Dosage error: {treatment['error']}")
            return flask.jsonify({"error": treatment['error']}), 400

        if disease.lower().find('malaria') != -1:
            counseling = {
                "disease_explanation": "Malaria is caused by parasites transmitted by mosquito bites.",
                "treatment_instructions": [
                    "Take the full course of treatment as prescribed.",
                    "Take medicines with food or fluids (especially Coartem).",
                    "If vomiting occurs within 30 minutes, repeat the dose.",
                    "Do not change treatment without consulting a health worker.",
                    "Return to a health facility if symptoms persist after 48 hours or worsen."
                ],
                "prevention": [
                    "Sleep under insecticide-treated mosquito nets.",
                    "Avoid mosquito bites by closing doors/windows early and screening houses.",
                    "Eliminate stagnant water to reduce mosquito breeding."
                ]
            }
        elif disease.lower().find('dengue') != -1:
            counseling = {
                "disease_explanation": "Dengue is a mosquito-borne viral illness.",
                "treatment_instructions": [
                    "Take paracetamol as prescribed, not aspirin or ibuprofen.",
                    "Drink plenty of fluids (water, oral rehydration salts).",
                    "Rest and avoid strenuous activity.",
                    "Return to a health facility if symptoms worsen (e.g., bleeding, severe pain, confusion)."
                ],
                "prevention": [
                    "Use insect repellent (e.g., DEET) and wear long clothing.",
                    "Sleep under insecticide-treated bed nets, especially during the day.",
                    "Eliminate stagnant water around the home to reduce mosquito breeding."
                ]
            }
        elif disease.lower().find('typhoid') != -1:
            counseling = {
                "disease_explanation": "Typhoid is a bacterial infection from contaminated food or water.",
                "treatment_instructions": [
                    "Complete the full antibiotic course (14 days).",
                    "Take ciprofloxacin with food to reduce stomach upset.",
                    "Avoid alcohol during treatment.",
                    "Return if fever persists >72 hours or new symptoms (e.g., severe abdominal pain) appear."
                ],
                "prevention": [
                    "Drink safe, clean water (boiled or treated).",
                    "Wash hands before eating and after defecation.",
                    "Ensure proper food hygiene (cook food thoroughly)."
                ]
            }
        else:
            counseling = {
                "disease_explanation": "Consult a health worker for accurate diagnosis.",
                "treatment_instructions": ["Follow health worker advice for treatment."],
                "prevention": ["Seek medical advice for prevention strategies."]
            }

        logger.info(f"Prediction: Disease={disease}, Treatment={treatment}")
        return flask.jsonify({
            'prediction': {
                'disease': disease,
                'treatment': treatment,
                'counseling': counseling
            }
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return flask.jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return flask.jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

# Define valid values
VALID_REGIONS = [
    "Central America west of Panama", "Haiti", "Dominican Republic",
    "Sub-Saharan Africa", "Papua New Guinea", "Southeast Asia"
]
VALID_GENDERS = ['Male', 'Female']
VALID_PREV_MEDS = []
VALID_SYMPTOMS = [
    'fever', 'chills', 'sweats', 'headache', 'nausea', 'vomiting', 'body_aches',
    'impaired_consciousness', 'prostration', 'convulsions', 'deep_breathing',
    'respiratory_distress', 'abnormal_bleeding', 'jaundice', 'severe_anemia',
    'rash', 'abdominal_pain', 'weakness', 'diarrhea', 'severe_dehydration',
    'constipation', 'delirium', 'mucosal_bleeding', 'lethargy'
]

# Dosage tables
COARTEM_DOSAGE = {
    (5, 14): {"tablets": 1, "frequency": "twice daily (12 hourly)", "duration": "3 days", "color": "Yellow"},
    (15, 24): {"tablets": 2, "frequency": "twice daily (12 hourly)", "duration": "3 days", "color": "Blue"},
    (25, 34): {"tablets": 3, "frequency": "twice daily (12 hourly)", "duration": "3 days", "color": "Brown"},
    (35, float('inf')): {"tablets": 4, "frequency": "twice daily (12 hourly)", "duration": "3 days", "color": "Green"}
}

ARTESUNATE_AMODIAQUINE_DOSAGE = {
    (5, 11): {
        "artesunate": {"mg": 25, "tablets": 0.5, "days": [1, 2, 3]},
        "amodiaquine": {"mg": 76, "tablets": 0.5, "days": [1, 2, 3]}
    },
    (12, 35): {
        "artesunate": {"mg": 50, "tablets": 1, "days": [1, 2, 3]},
        "amodiaquine": {"mg": 153, "tablets": 1, "days": [1, 2, 3]}
    },
    (36, 71): {
        "artesunate": {"mg": 100, "tablets": 2, "days": [1, 2, 3]},
        "amodiaquine": {"mg": 306, "tablets": 2, "days": [1, 2, 3]}
    },
    (72, float('inf')): {
        "artesunate": {"mg": 200, "tablets": 4, "days": [1, 2, 3]},
        "amodiaquine": {"mg": 612, "tablets": 4, "days": [1, 2, 3]}
    }
}

QUININE_DOSAGE = {
    (5, 9): {"mg": 75, "tablets": 0.25, "frequency": "every 8 hours", "duration": "7 days"},
    (10, 17): {"mg": 150, "tablets": 0.5, "frequency": "every 8 hours", "duration": "7 days"},
    (18, 23): {"mg": 225, "tablets": 0.75, "frequency": "every 8 hours", "duration": "7 days"},
    (24, 29): {"mg": 300, "tablets": 1, "frequency": "every 8 hours", "duration": "7 days"},
    (30, 39): {"mg": 375, "tablets": 1.25, "frequency": "every 8 hours", "duration": "7 days"},
    (40, 49): {"mg": 450, "tablets": 1.5, "frequency": "every 8 hours", "duration": "7 days"},
    (50, float('inf')): {"mg": 600, "tablets": 2, "frequency": "every 8 hours", "duration": "7 days"}
}

PARACETAMOL_DOSAGE = {
    (5, 14): {"mg": 125, "tablets": 0.25, "max_tablets_daily": 0.75},
    (15, 24): {"mg": 250, "tablets": 0.5, "max_tablets_daily": 1.5},
    (25, 34): {"mg": 500, "tablets": 1, "max_tablets_daily": 3},
    (35, 49): {"mg": 750, "tablets": 1.5, "max_tablets_daily": 4.5},
    (50, float('inf')): {"mg": 1000, "tablets": 2, "max_tablets_daily": 6}
}

DENGUE_PARACETAMOL_DOSAGE = {
    (5, 14): {"mg": 125, "tablets": 0.25, "max_tablets_daily": 0.75, "frequency": "every 8 hours if fever >=38.5°C"},
    (15, 24): {"mg": 250, "tablets": 0.5, "max_tablets_daily": 1.5, "frequency": "every 8 hours if fever >=38.5°C"},
    (25, 34): {"mg": 500, "tablets": 1, "max_tablets_daily": 3, "frequency": "every 8 hours if fever >=38.5°C"},
    (35, 49): {"mg": 750, "tablets": 1.5, "max_tablets_daily": 4.5, "frequency": "every 8 hours if fever >=38.5°C"},
    (50, float('inf')): {"mg": 1000, "tablets": 2, "max_tablets_daily": 6, "frequency": "every 8 hours if fever >=38.5°C"}
}

CIPROFLOXACIN_DOSAGE = {
    (5, 29): {"mg": 250, "tablets": 0.5, "frequency": "every 12 hours", "duration": "14 days", "route": "oral"},
    (30, float('inf')): {"mg": 500, "tablets": 1, "frequency": "every 12 hours", "duration": "14 days", "route": "oral"}
}

CEFTRIAXONE_DOSAGE = {
    (5, 29): {"mg": 500, "frequency": "every 12 hours", "duration": "14 days", "route": "IV"},
    (30, float('inf')): {"mg": 1000, "frequency": "every 12 hours", "duration": "14 days", "route": "IV"}
}

def get_dengue_dosage(weight, warning_signs, severe):
    if severe or warning_signs:
        return {"error": "Severe dengue or warning signs detected. Refer to hospital for IV fluids."}
    if weight < 5:
        return {"error": "Weight <5 kg; consult a pediatric specialist for dengue management."}
    for (min_w, max_w), dosage in DENGUE_PARACETAMOL_DOSAGE.items():
        if min_w <= weight <= max_w:
            return {
                "drug": "Paracetamol (500 mg)",
                "dosage": {
                    "mg_per_dose": dosage["mg"],
                    "tablets_per_dose": dosage["tablets"],
                    "frequency": dosage["frequency"],
                    "max_tablets_daily": dosage["max_tablets_daily"]
                },
                "instructions": "Ensure adequate hydration (2–3L/day for adults). Stop NSAIDs (e.g., aspirin).",
                "warnings": ["Seek immediate care if bleeding, severe pain, or lethargy develops."]
            }
    return {"error": "No Paracetamol dosage for this weight"}

def get_typhoid_dosage(weight, resistance=False):
    if weight < 5:
        return {"error": "Antibiotics not recommended for weight <5 kg. Refer to specialist."}
    drug = "Ceftriaxone (1 g)" if resistance else "Ciprofloxacin (500 mg)"
    dosage_table = CEFTRIAXONE_DOSAGE if resistance else CIPROFLOXACIN_DOSAGE
    for (min_w, max_w), dosage in dosage_table.items():
        if min_w <= weight <= max_w:
            return {
                "drug": drug,
                "dosage": dosage,
                "supportive": get_paracetamol_dosage(weight) if weight >= 5 else None,
                "instructions": "Complete full antibiotic course. Take ciprofloxacin with food.",
                "warnings": ["Stop NSAIDs (e.g., aspirin). Seek care if fever persists >72 hours or abdominal pain worsens."]
            }
    return {"error": f"No {drug} dosage for this weight"}

def get_dosage(weight, drug, pregnant, first_trimester, g6pd_deficiency):
    if weight < 5:
        logger.warning(f"Weight {weight} kg is below 5 kg; recommending Quinine")
        for (min_w, max_w), dosage in QUININE_DOSAGE.items():
            if min_w <= weight <= max_w:
                return {
                    "drug": "Quinine",
                    "dosage": dosage,
                    "warnings": ["Patient weight <5 kg; ACTs contraindicated."],
                    "supportive": get_paracetamol_dosage(weight)
                }
        return {"error": "No Quinine dosage for this weight"}
    if pregnant and first_trimester:
        logger.info("First trimester pregnancy detected; recommending Quinine")
        for (min_w, max_w), dosage in QUININE_DOSAGE.items():
            if min_w <= weight <= max_w:
                return {
                    "drug": "Quinine",
                    "dosage": dosage,
                    "warnings": ["First trimester pregnancy; ACTs contraindicated."],
                    "supportive": get_paracetamol_dosage(weight)
                }
        return {"error": "No Quinine dosage for this weight"}
    for (min_w, max_w), dosage in COARTEM_DOSAGE.items():
        if min_w <= weight <= max_w:
            warnings = []
            if g6pd_deficiency:
                warnings.append("G6PD deficiency noted; Amodiaquine not recommended as alternative.")
            return {
                "drug": "Artemether/Lumefantrine (Coartem)",
                "dosage": dosage,
                "alternative": "Artesunate + Amodiaquine (if Coartem unavailable, avoid if G6PD deficient)",
                "warnings": warnings,
                "supportive": get_paracetamol_dosage(weight)
            }
    return {"error": "No Coartem dosage for this weight"}

def get_paracetamol_dosage(weight):
    if weight < 5:
        return {"note": "Paracetamol not recommended for weight <5 kg"}
    for (min_w, max_w), dosage in PARACETAMOL_DOSAGE.items():
        if min_w <= weight <= max_w:
            return {
                "drug": "Paracetamol (500 mg)",
                "dosage": {
                    "mg_per_dose": dosage["mg"],
                    "tablets_per_dose": dosage["tablets"],
                    "frequency": "every 8 hours if fever >=38.5°C",
                    "max_tablets_daily": dosage["max_tablets_daily"]
                },
                "instructions": "Use with tepid sponging for high fever."
            }
    return {"note": "No Paracetamol dosage for this weight"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)