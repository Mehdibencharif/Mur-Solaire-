import os
import numpy as np
import pandas as pd
import pydeck as pdk
from urllib.parse import quote_plus
from geopy.geocoders import Nominatim
import streamlit as st


# ==========================
# SECTION 1 – LOCALISATION & ORIENTATION (VISUEL + AZIMUT)
# ==========================

st.header("1) Localisation & Orientation")

# --- Adresse + liens rapides ---
adresse = st.text_input(
    "Adresse du site (ou point d’intérêt)",
    value=st.session_state.get("adresse", "Saint-Augustin-de-Desmaures, QC"),
    help="Ex.: 'Usine ABC, 123 rue X, Ville' ou 'Code postal'."
)
st.session_state["adresse"] = adresse

q = quote_plus(adresse.strip()) if adresse.strip() else ""
lien_maps  = f"https://www.google.com/maps/search/?api=1&query={q}" if q else ""
lien_earth = f"https://earth.google.com/web/search/{q}" if q else ""

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"🔗 **Google Maps** : [{('Ouvrir dans Maps' if q else '—')}]({lien_maps})" if q else "🔗 **Google Maps** : —")
with c2:
    st.markdown(f"🌍 **Google Earth** : [{('Ouvrir dans Earth' if q else '—')}]({lien_earth})" if q else "🌍 **Google Earth** : —")

# --- Géocodage adresse -> lat/lon (avec repli manuel) ---
@st.cache_data(show_spinner=False)
def geocode_addr(addr: str):
    if not addr or not addr.strip():
        return None
    try:
        geolocator = Nominatim(user_agent="mur_solaire_app")
        loc = geolocator.geocode(addr, timeout=10)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        pass
    return None

coords = geocode_addr(adresse) if adresse else None

# Valeurs par défaut Québec (si géocodage indisponible)
default_lat, default_lon = 46.813900, -71.208000

colA, colB = st.columns(2)
with colA:
    lat = st.number_input(
        "Latitude",
        value=float(coords[0]) if coords else float(st.session_state.get("lat", default_lat)),
        format="%.6f"
    )
with colB:
    lon = st.number_input(
        "Longitude",
        value=float(coords[1]) if coords else float(st.session_state.get("lon", default_lon)),
        format="%.6f"
    )

# Sauvegarde persistante
st.session_state["lat"] = float(lat)
st.session_state["lon"] = float(lon)

if not coords and adresse.strip():
    st.warning("Géocodage indisponible ou infructueux. Coordonnées par défaut affichées — ajuste-les au besoin.")

with st.expander("Comment mesurer/valider l’azimut ?"):
    st.write(
        "- L’**azimut** est mesuré **depuis le Nord**, en degrés et **sens horaire**.\n"
        "- **0°** = Nord, **90°** = Est, **180°** = Sud, **270°** = Ouest.\n"
        "- ⚠️ La valeur **ne doit jamais être négative** (0–359.99°).\n"
        "- Astuce : dans Google Maps, utilise l’outil **règle** et compare avec le Nord."
    )

# --- Orientation & conditions visuelles ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    azimuth = st.number_input(
        "Azimut du mur (°)", value=float(st.session_state.get("azimuth", 151.22)),
        min_value=0.0, max_value=359.99, step=0.01,
        help="0–359.99°, depuis le Nord (sens horaire). Exemple: 151° ≈ Sud-Sud-Est."
    )
with col2:
    tilt = st.number_input(
        "Inclinaison (°)", value=float(st.session_state.get("tilt", 90.0)),
        min_value=0.0, max_value=90.0, step=1.0,
        help="0° = horizontal (toit), 90° = vertical (façade)."
    )
with col3:
    shading = st.slider(
        "Ombrage global (%)", min_value=0, max_value=90, value=int(st.session_state.get("shading", 10)), step=1,
        help="Pertes d’irradiation dues aux obstacles proches/lointains."
    )
with col4:
    wind_ref = st.number_input(
        "Vent (m/s – indicatif)", value=float(st.session_state.get("wind_ref", 3.0)),
        min_value=0.0, step=0.5,
        help="Valeur indicative pour contexte (non utilisée dans les calculs MVP)."
    )

st.session_state.update({"azimuth": float(azimuth), "tilt": float(tilt), "shading": int(shading), "wind_ref": float(wind_ref)})

# --- Petit utilitaire: azimut -> point cardinal
def azimut_cardinal(a):
    labels = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSO","SO","OSO","O","ONO","NO","NNO"]
    idx = int((a % 360) / 22.5 + 0.5) % 16
    return labels[idx]

st.caption(
    f"🧭 **Azimut** : {azimuth:.2f}° ({azimut_cardinal(azimuth)}) • "
    f"📐 **Inclinaison** : {tilt:.0f}° • "
    f"🌫️ **Ombrage** : {shading}% • "
    f"💨 **Vent** : {wind_ref:.1f} m/s"
)

# --- Carte PyDeck avec flèche d’azimut ---
def destination_point(lat_deg, lon_deg, bearing_deg, distance_m=200.0):
    R = 6371000.0
    br = np.deg2rad(bearing_deg)
    lat1 = np.deg2rad(lat_deg); lon1 = np.deg2rad(lon_deg)
    lat2 = np.arcsin(np.sin(lat1)*np.cos(distance_m/R) + np.cos(lat1)*np.sin(distance_m/R)*np.cos(br))
    lon2 = lon1 + np.arctan2(np.sin(br)*np.sin(distance_m/R)*np.cos(lat1),
                             np.cos(lat1)*np.cos(distance_m/R) - np.sin(lat1)*np.sin(lat2))
    return np.rad2deg(lat2), np.rad2deg(lon2)

end_lat, end_lon = destination_point(lat, lon, azimuth, 200.0)

point_df = pd.DataFrame([{"lat": lat, "lon": lon}])
line_df = pd.DataFrame([{"lat": lat, "lon": lon}, {"lat": end_lat, "lon": end_lon}])

site_layer = pdk.Layer(
    "ScatterplotLayer",
    data=point_df,
    get_position='[lon, lat]',
    get_radius=6,
    radius_scale=10,
    pickable=True,
)
arrow_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": line_df[["lon","lat"]].values.tolist()}],
    get_width=4,
    width_min_pixels=2,
    pickable=False,
)

view_state = pdk.ViewState(
    longitude=lon, latitude=lat, zoom=15, pitch=45, bearing=float(azimuth)
)

tooltip = {"html": "<b>Site</b><br/>Lat: {lat}<br/>Lon: {lon}", "style": {"color": "white"}}
mapbox_key = os.getenv("MAPBOX_API_KEY", "")

if mapbox_key:
    pdk.settings.mapbox_api_key = mapbox_key
    deck = pdk.Deck(
        map_provider="mapbox",
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[site_layer, arrow_layer],
        tooltip=tooltip,
    )
else:
    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=[site_layer, arrow_layer],
        tooltip=tooltip,
    )

st.pydeck_chart(deck, use_container_width=True)


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










