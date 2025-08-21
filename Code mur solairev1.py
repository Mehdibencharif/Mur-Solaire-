import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pydeck as pdk
from urllib.parse import quote_plus
from geopy.geocoders import Nominatim

# ==========================
# CONFIG APP
# ==========================
st.set_page_config(page_title="Mur solaire – Audit Flash", layout="wide")
st.title("Audit Flash – Mur solaire")
st.caption("V1.0 – Prototype : estimation simple des gains thermiques, coûts, subventions et rentabilité. Basé sur des entrées clés inspirées de RETScreen.")

# ==========================
# SECTION 1 bis – PARAMÈTRES CLIMATIQUES (style RETScreen)
# ==========================
st.subheader("Paramètres climatiques (RETScreen-like)")

colh1, colh2, colh3 = st.columns(3)
with colh1:
    zone_clim = st.selectbox(
        "Zone climatique",
        options=["1 - Très chaud","2 - Chaud","3 - Tempéré chaud","4 - Tempéré",
                 "5 - Tempéré froid","6 - Froid","7 - Très froid","8 - Arctique"],
        index=6,  # "7 - Très froid" par défaut
        help="Classification indicative pour le dimensionnement."
    )
with colh2:
    elevation_m = st.number_input("Élévation (m)", value=75.0, step=1.0)
with colh3:
    amp_sol = st.number_input("Amplitude des T° du sol (°C)", value=24.2, step=0.1)

colt1, colt2, colt3 = st.columns(3)
with colt1:
    t_ext_chauff = st.number_input("T° ext. de calcul (chauffage) (°C)", value=-23.6, step=0.1)
with colt2:
    t_ext_clim = st.number_input("T° ext. de calcul (climatisation) (°C)", value=27.3, step=0.1)
with colt3:
    vent_ref = st.number_input("Vitesse du vent réf. (m/s)", value=4.0, step=0.1)

# --------------------------
# PRÉREMPLISSAGE AVANT L'ÉDITEUR
# --------------------------
st.markdown("### Source des données climatiques")

source_climat = st.radio(
    "Choisir la source :",
    ["Manuel", "Préréglage local (SADM – valeurs type RETScreen)", "Auto (calcul DD à partir des T°)"],
    index=1,
    help="Par défaut: préréglage embarqué. Modifiable ensuite."
)

# Jeu local embarqué (proche de ta capture)
DEFAULT_CLIMATE_SADM = {
    "Mois": ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"],
    "Temp. air (°C)": [-12.4, -11.0, -4.6, 3.3, 10.8, 16.3, 19.1, 17.2, 12.5, 6.5, 0.5, -9.1],
    "HR (%)": [69.1, 66.8, 66.1, 64.4, 64.0, 68.8, 73.6, 74.1, 75.9, 74.1, 74.1, 75.0],
    "Précip. (mm)": [68.29, 64.52, 79.27, 81.89, 96.29, 119.33, 122.19, 114.88, 102.99, 112.61, 101.26, 92.38],
    "Rayon. horiz. (kWh/m²/j)": [1.62, 2.66, 3.92, 4.92, 5.76, 5.30, 5.65, 4.43, 3.49, 2.61, 1.85, 1.52],
    "Pression (kPa)": [100.6, 100.6, 100.5, 100.5, 100.6, 100.5, 100.4, 100.5, 100.7, 100.8, 100.7, 100.7],
    "Vent (m/s)": [4.7, 4.7, 4.7, 4.5, 4.2, 3.6, 3.1, 3.4, 3.3, 3.9, 4.3, 4.5],
    "T° sol (°C)": [-14.6, -12.7, -6.7, 2.5, 10.0, 16.8, 19.0, 18.2, 13.0, 5.4, -1.9, -10.3],
    # colonnes DD vides: seront calculées
    "DD18 (°C·j)": [None]*12,
    "DD10 (°C·j)": [None]*12,
}

import calendar
import numpy as np
import pandas as pd

def compute_degree_days(df, base_heat=18.0, base_cool=10.0, year=2024):
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    T = np.asarray(df["Temp. air (°C)"], dtype=float)
    hdd18 = np.maximum(0.0, base_heat - T) * days
    cdd10 = np.maximum(0.0, T - base_cool) * days
    out = df.copy()
    out["DD18 (°C·j)"] = np.round(hdd18, 0)
    out["DD10 (°C·j)"] = np.round(cdd10, 0)
    return out

# 1) Construire le DataFrame initial selon la source
if "climat_mensuel_df" not in st.session_state:
    st.session_state["climat_mensuel_df"] = pd.DataFrame(DEFAULT_CLIMATE_SADM)

if source_climat == "Préréglage local (SADM – valeurs type RETScreen)":
    base_df = pd.DataFrame(DEFAULT_CLIMATE_SADM)
    base_df = compute_degree_days(base_df)
elif source_climat == "Auto (calcul DD à partir des T°)":
    # Part de l'état en session (modifiable), DD recalculés à chaque run
    base_df = compute_degree_days(st.session_state["climat_mensuel_df"])
else:  # Manuel
    # Gabarit vide si tu veux repartir de zéro
    mois = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    base_df = pd.DataFrame({
        "Mois": mois,
        "Temp. air (°C)": [np.nan]*12,
        "HR (%)": [np.nan]*12,
        "Précip. (mm)": [np.nan]*12,
        "Rayon. horiz. (kWh/m²/j)": [np.nan]*12,
        "Pression (kPa)": [np.nan]*12,
        "Vent (m/s)": [np.nan]*12,
        "T° sol (°C)": [np.nan]*12,
        "DD18 (°C·j)": [np.nan]*12,
        "DD10 (°C·j)": [np.nan]*12,
    })

# 2) Éditeur (maintenant il s’ouvre déjà prérempli)
clim_df = st.data_editor(
    base_df,
    key="clim_editor",
    num_rows="fixed",
    column_config={
        "Temp. air (°C)": st.column_config.NumberColumn(format="%.1f"),
        "HR (%)": st.column_config.NumberColumn(format="%.1f"),
        "Précip. (mm)": st.column_config.NumberColumn(format="%.2f"),
        "Rayon. horiz. (kWh/m²/j)": st.column_config.NumberColumn(format="%.2f"),
        "Pression (kPa)": st.column_config.NumberColumn(format="%.1f"),
        "Vent (m/s)": st.column_config.NumberColumn(format="%.1f"),
        "T° sol (°C)": st.column_config.NumberColumn(format="%.1f"),
        "DD18 (°C·j)": st.column_config.NumberColumn(format="%.0f"),
        "DD10 (°C·j)": st.column_config.NumberColumn(format="%.0f"),
    },
    use_container_width=True,
    hide_index=True,
)

# 3) Si la source est "Auto", recalcule DD après édition (live)
if source_climat == "Auto (calcul DD à partir des T°)":
    clim_df = compute_degree_days(clim_df)

# 4) Sauvegarde pour persistance
st.session_state["climat_mensuel_df"] = clim_df
st.session_state["climat_meta"] = {
    "latitude": float(lat),
    "longitude": float(lon),
    "zone_climatique": zone_clim,
    "elevation_m": elevation_m,
    "t_ext_calc_chauffage_C": t_ext_chauff,
    "t_ext_calc_clim_C": t_ext_clim,
    "amplitude_sol_C": amp_sol,
    "vent_ref_ms": vent_ref,
}

with st.expander("Synthèse annuelle"):
    moy_air = clim_df["Temp. air (°C)"].mean(skipna=True)
    moy_vent = clim_df["Vent (m/s)"].mean(skipna=True)
    moy_ray = clim_df["Rayon. horiz. (kWh/m²/j)"].mean(skipna=True)
    sum_dd18 = clim_df["DD18 (°C·j)"].sum(skipna=True)
    sum_dd10 = clim_df["DD10 (°C·j)"].sum(skipna=True)
    st.write(
        f"• **T° air moyenne**: {moy_air:.1f} °C | "
        f"**Vent moyen**: {moy_vent:.1f} m/s | "
        f"**Rayonnement moyen**: {moy_ray:.2f} kWh/m²/j | "
        f"**DD18 annuels**: {sum_dd18:.0f} °C·j | "
        f"**DD10 annuels**: {sum_dd10:.0f} °C·j"
    )

st.success("Paramètres climatiques préremplis et enregistrés.")

# ==========================
# SECTION 2 – CLIMAT & ENERGIE SOLAIRE INCIDENTE
# ==========================
st.header("2) Climat & irradiation sur le plan du mur")
mode_meteo = st.radio("Source d’irradiation (kWh/m²·an ou mensuel)", ["Saisie rapide (annuelle)", "Tableau mensuel (upload RETScreen .csv/.xlsx)"]) 

annual_kwh_m2 = None
monthly_df = None

if mode_meteo == "Saisie rapide (annuelle)":
    annual_kwh_m2 = st.number_input("Irradiation annuelle sur plan du mur (kWh/m²·an)", value=350.0, min_value=50.0, max_value=1200.0, step=10.0)
    st.caption("Astuce : pour un mur orienté SSE/SE au Québec, une valeur d’ordre 300–500 kWh/m²·an sur plan vertical est courante. Remplacer par vos données RETScreen si disponibles.")
else:
    up = st.file_uploader("Importer un fichier mensuel RETScreen (colonnes Mois, kWh/m²)", type=["csv", "xlsx"]) 
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                monthly_df = pd.read_csv(up)
            else:
                monthly_df = pd.read_excel(up)
            # Normalisation colonnes
            monthly_df.columns = [c.strip().lower() for c in monthly_df.columns]
            # Recherche colonnes
            mcol, valcol = None, None
            for c in monthly_df.columns:
                if ("mois" in c) or ("month" in c):
                    mcol = c
                if "kwh" in c:
                    valcol = c
            if mcol is None or valcol is None:
                st.error("Le fichier doit contenir une colonne Mois et une colonne d’irradiation (kWh/m²).")
            else:
                monthly_df = monthly_df[[mcol, valcol]].copy()
                monthly_df.columns = ["Mois", "kWh/m²"]

                # Tentative de tri correct des mois si 12 lignes
                mois_ordre = ["jan", "fév", "fev", "mar", "avr", "mai", "jun", "jui", "aoû", "aou", "sep", "oct", "nov", "déc", "dec"]
                if len(monthly_df) == 12:
                    def key_mois(x):
                        s = str(x).strip().lower()[:3]
                        for i, m in enumerate(mois_ordre):
                            if s == m:
                                return i
                        return 99
                    monthly_df["__k"] = monthly_df["Mois"].apply(key_mois)
                    monthly_df = monthly_df.sort_values("__k").drop(columns="__k")

                annual_kwh_m2 = float(monthly_df["kWh/m²"].sum())
                st.success(f"Irradiation annuelle reconstituée : {annual_kwh_m2:,.0f} kWh/m²·an")
                # Graphique
                fig = plt.figure(figsize=(6,3))
                plt.bar(monthly_df["Mois"].astype(str), monthly_df["kWh/m²"])
                plt.ylabel("kWh/m²")
                plt.title("Irradiation mensuelle sur le plan du mur")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lecture fichier : {e}")

# ==========================
# SECTION 3 – PERFORMANCE COLLECTEUR (UTC / mur solaire)
# ==========================
st.header("3) Performance – Mur solaire (transpiré non vitré)")
colp1, colp2, colp3 = st.columns(3)
with colp1:
    eta0 = st.number_input("Rendement nominal η₀ (fraction)", value=0.65, min_value=0.1, max_value=0.9, step=0.01)
with colp2:
    sys_derate = st.number_input("Pertes système (ventilateur, fuites, etc.) %", value=5.0, min_value=0.0, max_value=20.0, step=0.5)
with colp3:
    frac_saison = st.slider("Part de l’irradiation utile (chauffe) %", min_value=20, max_value=100, value=70, step=5)

# Formule simple : Q_util (kWh/an) = A(m²) * G_an (kWh/m²·an) * η₀ * (1 - ombrage) * (1 - derating) * part_saison * (disponibilité)
if annual_kwh_m2 is not None:
    A = area_m2
    G = annual_kwh_m2
    ombrage = shading/100.0
    derate = sys_derate/100.0
    saison = frac_saison/100.0
    dispo = avail/100.0

    q_util_kwh = A * G * eta0 * (1 - ombrage) * (1 - derate) * saison * dispo
    st.subheader("🔸 Résultat – Chaleur utile estimée")
    st.metric("Q utile (kWh/an)", f"{q_util_kwh:,.0f}")
else:
    st.info("Saisir ou importer l’irradiation pour calculer la chaleur utile.")

# ==========================
# SECTION 4 – SUBSTITUTION ÉNERGÉTIQUE & ÉCONOMIES
# ==========================
st.header("4) Substitution énergétique & économies")
energie_cible = st.selectbox("Énergie remplacée principalement", ["Gaz naturel", "Électricité", "Autre (kWh équivalent)"])
colc1, colc2, colc3 = st.columns(3)
with colc1:
    prix_gaz_kwh = st.number_input("Prix gaz naturel ($/kWh PCI)", value=0.05, format="%.3f")
with colc2:
    prix_el_kwh = st.number_input("Prix électricité ($/kWh)", value=0.10, format="%.3f")
with colc3:
    rendement_chauffage = st.number_input("Rendement chauffage existant (%)", value=85.0, min_value=40.0, max_value=100.0, step=1.0)

# Init par défaut pour éviter références avant assignation
kwh_final_evit = 0.0
eco_dollars = 0.0
ges_tonnes = 0.0

if annual_kwh_m2 is not None:
    rdt = max(rendement_chauffage/100.0, 1e-6)
    if energie_cible == "Gaz naturel":
        val_kwh = prix_gaz_kwh
        ges_factor = co2_kg_per_kwh_ng
    elif energie_cible == "Électricité":
        val_kwh = prix_el_kwh
        ges_factor = co2_kg_per_kwh_el
    else:
        val_kwh = st.number_input("Tarif ($/kWh équivalent)", value=0.07, format="%.3f")
        ges_factor = st.number_input("Facteur GES (kg CO₂e/kWh)", value=0.100, format="%.3f")

    # L’énergie solaire utile remplace l’énergie finale / le rendement du système remplacé
    kwh_final_evit = q_util_kwh / rdt
    eco_dollars = kwh_final_evit * val_kwh
    ges_tonnes = (kwh_final_evit * ges_factor) / 1000.0

    met1, met2, met3 = st.columns(3)
    met1.metric("Énergie finale évitée (kWh/an)", f"{kwh_final_evit:,.0f}")
    met2.metric("Économies annuelles (dollars/an)", f"{eco_dollars:,.0f}")
    met3.metric("GES évités (t CO₂e/an)", f"{ges_tonnes:,.2f}")

# ==========================
# SECTION 5 – COÛTS, MARGE & SUBVENTIONS
# ==========================
st.header("5) Coûts, marge & subventions")
colk1, colk2, colk3 = st.columns(3)
with colk1:
    cout_mat_pi2 = st.number_input("Matériaux ($/pi²)", value=24.0, step=1.0)
    cout_mo_pi2 = st.number_input("Main-d'œuvre ($/pi²)", value=12.0, step=1.0)
with colk2:
    autres_fixes = st.number_input("Autres coûts fixes ($)", value=0.0, step=500.0)
    marge_pct = st.number_input("Marge (%)", value=20.0, min_value=0.0, max_value=50.0, step=1.0)
with colk3:
    sub_type = st.selectbox("Type de subvention", ["Aucune", "% du CAPEX", "$ par m² (plafonné)"])

capex_base = area_ft2 * (cout_mat_pi2 + cout_mo_pi2) + autres_fixes
marge = capex_base * (marge_pct/100.0)
capex_avant_sub = capex_base + marge

# Subventions
sub_amount = 0.0
if sub_type == "% du CAPEX":
    sub_pct = st.number_input("Subvention (% du CAPEX)", value=30.0, min_value=0.0, max_value=90.0, step=1.0)
    sub_amount = capex_avant_sub * (sub_pct/100.0)
elif sub_type == "$ par m² (plafonné)":
    sub_per_m2 = st.number_input("$ par m²", value=150.0, step=10.0)
    sub_cap = st.number_input("Plafond de subvention ($)", value=250000.0, step=5000.0)
    sub_amount = min(area_m2 * sub_per_m2, sub_cap)

capex_net = max(capex_avant_sub - sub_amount, 0.0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("CAPEX (base)", f"{capex_base:,.0f} $")
k2.metric("Marge", f"{marge:,.0f} $")
k3.metric("Subvention estimée", f"{sub_amount:,.0f} $")
k4.metric("Investissement net", f"{capex_net:,.0f} $")

# ==========================
# SECTION 6 – INDICATEURS FINANCIERS
# ==========================
st.header("6) Indicateurs financiers")
colf1, colf2, colf3 = st.columns(3)
with colf1:
    years = st.number_input("Horizon d’analyse (ans)", min_value=1, max_value=30, value=15, step=1)
with colf2:
    discount = st.number_input("Taux d’actualisation (%)", value=6.0, min_value=0.0, max_value=20.0, step=0.5)
with colf3:
    escal = st.number_input("Escalade prix énergie (%/an)", value=2.0, min_value=0.0, max_value=15.0, step=0.5)

# Init pour export
npv_savings = 0.0
npv = -capex_net
spb = np.inf

if annual_kwh_m2 is not None and eco_dollars > 0:
    r = discount/100.0
    g = escal/100.0
    # flux d’économies croissantes : S0=eco_$, croissance g, actualisation r
    t = np.arange(1, years+1)
    savings_nominal = eco_dollars * ((1+g)**(t-1))
    discount_factors = 1 / ((1+r)**t)
    npv_savings = float(np.sum(savings_nominal * discount_factors))
    npv = npv_savings - capex_net
    spb = capex_net / eco_dollars if eco_dollars > 0 else np.inf

    f1, f2, f3 = st.columns(3)
    f1.metric("SPB simple (ans)", f"{spb:,.1f}" if np.isfinite(spb) else "∞")
    f2.metric("VAN des économies ($)", f"{npv_savings:,.0f}")
    f3.metric("VAN projet ($)", f"{npv:,.0f}")

    # Courbe économies actualisées
    cum_disc = np.cumsum(savings_nominal*discount_factors) - capex_net
    fig2 = plt.figure(figsize=(6,3))
    plt.plot(t, cum_disc)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Années")
    plt.ylabel("VAN cumulée ($)")
    plt.title("VAN cumulée – point mort")
    plt.tight_layout()
    st.pyplot(fig2)
elif annual_kwh_m2 is not None:
    st.info("Complète la section 4 pour calculer VAN/SPB (énergie remplacée et tarifs).")

# ==========================
# EXPORT RAPPORT
# ==========================
st.header("7) Export – Résumé Excel")
if annual_kwh_m2 is not None:
    out = BytesIO()
    resume = {
        "Surface_m2": [area_m2],
        "Azimut_deg": [azimuth],
        "Irradiation_kWh_m2_y": [annual_kwh_m2],
        "Rendement_eta0": [eta0],
        "Ombrage_%": [shading],
        "Disponibilite_%": [avail],
        "Part_saison_%": [frac_saison],
        "Q_utile_kWh_y": [q_util_kwh],
        "Energie_finale_evitee_kWh_y": [kwh_final_evit],
        "Economies_$_y": [eco_dollars],
        "GES_tCO2e_y": [ges_tonnes],
        "CAPEX_base_$": [capex_base],
        "Marge_$": [marge],
        "Subvention_$": [sub_amount],
        "CAPEX_net_$": [capex_net],
        "SPB_ans": [spb if np.isfinite(spb) else None],
        "VAN_savings_$": [npv_savings],
        "VAN_projet_$": [npv]
    }
    df_out = pd.DataFrame(resume)
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Résumé")
    st.download_button("📥 Télécharger le résumé Excel", data=out.getvalue(),
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       file_name="mur_solaire_audit_flash.xlsx")
else:
    st.info("Renseigner l’irradiation pour activer l’export.")

st.caption("⚠️ MVP pédagogique : à valider et étalonner avec RETScreen/mesures réelles (rendement, climat, périodes de fonctionnement, pertes spécifiques site).")
# Calcul









