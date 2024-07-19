import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random


# for custom CSS styling
with open("src/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load testing data
test_cases = pd.read_csv(
    "C:/Users/as/Desktop/Breast Cancer/breast_cancer_prediction/datasets/test_cases.csv"
)


def generate_random_sample():
    idx = random.choice(test_cases.index)
    sample = test_cases.iloc[idx]

    st.session_state["nras"] = sample["nras"]
    st.session_state["rasgef1b_mut"] = sample["rasgef1b_mut"]
    st.session_state["erbb3"] = sample["erbb3"]
    st.session_state["nottingham_prognostic_index"] = sample[
        "nottingham_prognostic_index"
    ]

    st.session_state["hsd17b8"] = sample["hsd17b8"]
    st.session_state["cohort"] = sample["cohort"]
    st.session_state["overall_survival_months"] = sample["overall_survival_months"]
    st.session_state["jak1"] = sample["jak1"]

    st.session_state["palld"] = sample["palld"]
    st.session_state["rps6"] = sample["rps6"]
    st.session_state["spry2"] = sample["spry2"]
    st.session_state["age_at_diagnosis"] = sample["age_at_diagnosis"]

    st.success(f"Original Label: {'Alive' if sample['overall_survival'] else 'Dead'}")


# Load scaler
with open(
    "C:/Users/as/Desktop/Breast Cancer/breast_cancer_prediction/models/scaler.pkl", "rb"
) as f:
    scaler = pickle.load(f)


# Load model
with open(
    "C:/Users/as/Desktop/Breast Cancer/breast_cancer_prediction/models/xgboost_model.pkl",
    "rb",
) as f:
    model = pickle.load(f)


st.title("Breast Cancer Status Prediction")

pressed = st.button("Generate Random Sample")
if pressed:
    generate_random_sample()


with st.form("Prediciton From"):

    col1, col2 = st.columns(2)

    with col1:
        nras = st.number_input(
            label="Neuroblastoma RAS (NRAS)",
            help="NRAS is a gene that encodes a protein involved in cell signal transduction, which is crucial for cell proliferation and survival. It is part of the RAS gene family.",
            key="nras",
        )
        rasgef1b_mut = st.number_input(
            label="Mutation in RASGEF1B",
            help="Ras Guanine Nucleotide Exchange Factor 1B: is involved in activating RAS proteins by facilitating the exchange of GDP for GTP. Mutations in RASGEF1B could affect the activation of RAS pathways, impacting cell growth and differentiation. This can contribute to oncogenesis if the signaling pathways are dysregulated.",
            min_value=0,
            max_value=1,
            step=1,
            key="rasgef1b_mut",
        )
        erbb3 = st.number_input(
            label="ERBB3",
            help="Erb-B2 Receptor Tyrosine Kinase 3, also known as HER3, is a member of the epidermal growth factor receptor (EGFR) family, involved in cell growth and differentiation. Overexpression or mutations in ERBB3 are linked to breast cancer. It often partners with ERBB2 (HER2) to drive cancer progression.",
            key="erbb3",
        )
        nottingham_prognostic_index = st.number_input(
            label="Nottingham Prognostic Index (NPI)",
            help="It is used to determine prognosis following surgery for breast cancer. Its value is calculated using three pathological criteria: the size of the tumour; the number of involved lymph nodes; and the grade of the tumour.",
            min_value=1.0,
            max_value=7.0,
            key="nottingham_prognostic_index",
        )
        hsd17b8 = st.number_input(
            label="HSD17B8",
            help="Hydroxysteroid (17-beta) Dehydrogenase 8, is involved in steroid metabolism, particularly in the conversion of active steroids to their inactive forms.",
            key="hsd17b8",
        )
        cohort = st.number_input(
            label="Cohort",
            help="Cohort is a group of subjects who share a defining characteristic (It takes a value from 1 to 5)",
            min_value=1,
            max_value=5,
            step=1,
            key="cohort",
        )

    with col2:
        overall_survival_months = st.number_input(
            label="Overall Survival Months",
            help="Total number of months that the patient live since she diagnosed as a breast cancer patient",
            min_value=0,
            key="overall_survival_months",
        )
        jak1 = st.number_input(
            label="JAK1",
            help="Janus Kinase 1, is a tyrosine kinase involved in the JAK-STAT signaling pathway, which transmits signals from various cytokines and growth factors.",
            key="jak1",
        )
        palld = st.number_input(
            label="PALLD",
            help="PALLD is involved in the organization of the actin cytoskeleton and cell motility.",
            key="palld",
        )
        rps6 = st.number_input(
            label="Ribosomal Protein S6 (RPS6)",
            help="RPS6 is a component of the ribosome and plays a role in protein synthesis. It is regulated by the mTOR pathway, which controls cell growth and proliferation.",
            key="rps6",
        )
        spry2 = st.number_input(
            label="SPRY2",
            help="Sprouty RTK Signaling Antagonist 2, acts as a regulator of receptor tyrosine kinase (RTK) signaling pathways, inhibiting excessive cell signaling.",
            key="spry2",
        )
        age_at_diagnosis = st.number_input(
            label="Age at Diagnosis",
            help="The patient's age when she diagnosed as a breast cancer patient",
            min_value=10,
            max_value=100,
            step=1,
            key="age_at_diagnosis",
        )

    submitted = st.form_submit_button("Predict")

    if submitted:
        inputs = np.array(
            [
                [
                    nras,
                    rasgef1b_mut,
                    erbb3,
                    nottingham_prognostic_index,
                    hsd17b8,
                    cohort,
                    overall_survival_months,
                    jak1,
                    palld,
                    rps6,
                    spry2,
                    age_at_diagnosis,
                ]
            ]
        )

        inputs = scaler.transform(inputs)

        prediction = model.predict(inputs)
        proba = model.predict_proba(inputs)

        if prediction[0] == 0:
            with col1:
                st.markdown("Model's Decision")
                st.info(f"This patient is dead")
            with col2:
                st.markdown(f"Model's Confidence")
                st.success(np.max(proba))

        elif prediction[0] == 1:
            with col1:
                st.markdown("Model's Decision")
                st.success(f"This patient is still living!")
            with col2:
                st.markdown(f"Model's Confidence")
                st.success(np.max(proba))

        graph = pd.DataFrame(
            {
                "label": ["Dead", "Alive"],
                "score": proba[0],
            }
        )
        st.bar_chart(
            graph, x="label", y="score", color="#4473ff", horizontal=True, height=300
        )
