import os
import math
import calendar
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk

from urllib.parse import quote_plus

# ----------------------------
# Config page
# ----------------------------
st.set_page_config(page_title="Mur solaire – Audit Flash", layout="wide")
st.title("Audit Flash – Mur solaire")
st.caption("v1.1 – 4 blocs consolidés : localisation/orientation/climat → capteur → coûts/économies → export.")

# ----------------------------
# Constantes & utilitaires
# ----------------------------
FT2_PER_M2 = 10.7639
CFM_PER_LPS = 2.11888

CO2_KG_PER_KWH_NG = 0.17942   # 179.42 g/kWh ~ GN PCI (réf. client)
CO2_KG_PER_KWH_QC = 0.00204    # 2.04 g/kWh ~ Hydro-Québec (mix très bas carbone)

MOIS_FR = [
    "Janvier","Février","Mars","Avril","Mai","Juin",
    "Juillet","Août","Septembre","Octobre","Novembre","Décembre"
]

# Conversions
m2_to_ft2 = lambda x: x * FT2_PER_M2
ft2_to_m2 = lambda x: x / FT2_PER_M2
lps_to_cfm = lambda x: x * CFM_PER_LPS
cfm_to_lps = lambda x: x / CFM_PER_LPS

# Azimut cardinal
CARD_16 = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S",
           "SSO","SO","OSO","O","ONO","NO","NNO"]

def azimut_cardinal(a: float) -> str:
    idx = int((a % 360) / 22.5 + 0.5) % 16
    return CARD_16[idx]

# Destination géodésique simple (approx sphérique)
def destination_point(lat_deg, lon_deg, bearing_deg, distance_m=200.0):
    R = 6371000.0
    br = np.deg2rad(bearing_deg)
    lat1 = np.deg2rad(lat_deg); lon1 = np.deg2rad(lon_deg)
    lat2 = np.arcsin(np.sin(lat1)*np.cos(distance_m/R) + np.cos(lat1)*np.sin(distance_m/R)*np.cos(br))
    lon2 = lon1 + np.arctan2(np.sin(br)*np.sin(distance_m/R)*np.cos(lat1),
                             np.cos(lat1)*np.cos(distance_m/R) - np.sin(lat1)*np.sin(lat2))
    return np.rad2deg(lat2), np.rad2deg(lon2)

# Degrés-jours à partir des T° moyennes mensuelles
def compute_degree_days(df, base_heat=18.0, base_cool=10.0, year=None):
    year = year or date.today().year
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    T = np.asarray(df["Temp. air (°C)"], dtype=float)
    out = df.copy()
    out["DD18 (°C·j)"] = np.round(np.maximum(0.0, base_heat - T) * days, 0)
    out["DD10 (°C·j)"] = np.round(np.maximum(0.0, T - base_cool) * days, 0)
    return out

# ==============================
# BLOC 1 – Localisation, Orientation & Climat
# ==============================
st.header("1) Localisation & Orientation")

# -- Adresse + liens
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

# -- Géocodage
@st.cache_data(show_spinner=False)
def geocode_addr(addr: str):
    if not addr or not addr.strip():
        return None
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="mur_solaire_app")
        loc = geolocator.geocode(addr, timeout=10)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        pass
    return None

coords = geocode_addr(adresse) if adresse else None
DEFAULT_LAT, DEFAULT_LON = 46.813900, -71.208000

colA, colB = st.columns(2)
with colA:
    lat = st.number_input(
        "Latitude",
        value=float(coords[0]) if coords else float(st.session_state.get("lat", DEFAULT_LAT)),
        format="%.6f"
    )
with colB:
    lon = st.number_input(
        "Longitude",
        value=float(coords[1]) if coords else float(st.session_state.get("lon", DEFAULT_LON)),
        format="%.6f"
    )

st.session_state["lat"], st.session_state["lon"] = float(lat), float(lon)

if not coords and adresse.strip():
    st.warning("Géocodage indisponible ou infructueux. Coordonnées par défaut affichées — ajuste-les au besoin.")

# -- Saisie directe (comme RETScreen)
with st.expander("Comment mesurer/valider l’azimut ?", expanded=False):
    st.write(
        "- L’**azimut** est mesuré **depuis le Nord**, en degrés **sens horaire**.\n"
        "- **0°** = Nord, **90°** = Est, **180°** = Sud, **270°** = Ouest.\n"
        "- Valeur **0–359.99°** (jamais négative)."
    )

def azimut_cardinal(a: float) -> str:
    labels = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSO","SO","OSO","O","ONO","NO","NNO"]
    return labels[int((a % 360) / 22.5 + 0.5) % 16]

col1, col2, col3, col4 = st.columns(4)
with col1:
    azimuth = st.number_input(
        "Azimut du mur (°)",
        value=float(st.session_state.get("azimuth", 151.22)),
        min_value=0.0, max_value=359.99, step=0.01,
        help="Saisie directe (méthode RETScreen). 151° ≈ Sud-Sud-Est."
    )
with col2:
    tilt = st.number_input(
        "Inclinaison (°)", value=float(st.session_state.get("tilt", 90.0)),
        min_value=0.0, max_value=90.0, step=1.0,
        help="0° = horizontal (toit), 90° = vertical (façade)."
    )
with col3:
    shading = st.slider(
        "Ombrage global (%)", min_value=0, max_value=90,
        value=int(st.session_state.get("shading", 10)), step=1,
        help="Pertes d’irradiation dues aux obstacles."
    )
with col4:
    wind_ref = st.number_input(
        "Vent (m/s – indicatif)", value=float(st.session_state.get("wind_ref", 3.0)),
        min_value=0.0, step=0.5
    )

st.session_state.update({
    "azimuth": float(azimuth),
    "tilt": float(tilt),
    "shading": int(shading),
    "wind_ref": float(wind_ref)
})

st.caption(
    f"🧭 **Azimut MUR** : {azimuth:.2f}° ({azimut_cardinal(azimuth)}) • "
    f"📐 **Inclinaison** : {tilt:.0f}° • "
    f"🌫️ **Ombrage** : {shading}% • "
    f"💨 **Vent** : {wind_ref:.1f} m/s"
)

# -- Flèche d’azimut sur carte (direction façade)
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
line_df  = pd.DataFrame([{"lat": lat, "lon": lon}, {"lat": end_lat, "lon": end_lon}])

site_layer  = pdk.Layer("ScatterplotLayer", data=point_df,
                        get_position='[lon, lat]', get_radius=6, radius_scale=10, pickable=True)
arrow_layer = pdk.Layer("PathLayer",
                        data=[{"path": line_df[["lon","lat"]].values.tolist()}],
                        get_width=4, width_min_pixels=2, pickable=False)

view_state = pdk.ViewState(longitude=lon, latitude=lat, zoom=15, pitch=45, bearing=float(azimuth))
mapbox_key = os.getenv("MAPBOX_API_KEY", "")
if mapbox_key:
    pdk.settings.mapbox_api_key = mapbox_key
    deck = pdk.Deck(map_provider="mapbox",
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state, layers=[site_layer, arrow_layer])
else:
    deck = pdk.Deck(map_provider="carto", map_style="light",
                    initial_view_state=view_state, layers=[site_layer, arrow_layer])

st.pydeck_chart(deck, use_container_width=True)

# ==============================
# BLOC 2 – Paramètres du capteur solaire à air
# ==============================
st.header("2) Paramètres du capteur solaire à air")

# ——— 2.0 Unités (globales à l’app, SI en interne) ———
st.radio("Unités", ["Métrique (SI)", "Impériales"], horizontal=True, key="unit_mode")
unit_mode = st.session_state.get("unit_mode", "Métrique (SI)")

# Conversions
FT2_PER_M2  = 10.7639
M2_PER_FT2  = 1.0 / FT2_PER_M2
CFM_PER_LPS = 2.11888
LPS_PER_CFM = 1.0 / CFM_PER_LPS
def m2_to_ft2(x): return x * FT2_PER_M2
def ft2_to_m2(x): return x * M2_PER_FT2
def lps_to_cfm(x): return x * CFM_PER_LPS
def cfm_to_lps(x): return x * LPS_PER_CFM

# ——— 2.1 Paramètres du capteur (type, surface, ombrage/vent) ———
with st.expander("Paramètres du capteur (type, surface, pertes)", expanded=True):
    TYPES = {
        "Mur solaire sans vitrage (UTSC)": {"absorptivite": 0.94, "facteur_correctif": 1.00,
            "comment": "Mur perforé aspiré (tirage méca). ΔT élevé par temps ensoleillé."},
        "Capteur à air vitré": {"absorptivite": 0.95, "facteur_correctif": 1.05,
            "comment": "Caisson vitré. Meilleur en intersaison; pertes nocturnes ↑."},
        "Vitré + absorbeur sélectif": {"absorptivite": 0.96, "facteur_correctif": 1.10,
            "comment": "Absorbeur sélectif; mieux à faible éclairement; coût ↑."},
    }
    type_capteur = st.selectbox("Type de capteur", list(TYPES.keys()), index=0)
    defaults = TYPES[type_capteur]

    colc1, colc2, colc3 = st.columns(3)
    absorptivite = colc1.number_input("Absorptivité du capteur", 0.80, 0.99, float(defaults["absorptivite"]), 0.01)
    couleur = colc1.selectbox("Couleur/finition", ["Noir", "Anthracite", "Autre"], index=0)

    facteur_correctif = colc2.number_input("Facteur correctif global (adim.)", 0.50, 2.00,
                                           float(defaults["facteur_correctif"]), 0.01,
                                           help="Calage global (ombrage résiduel, pertes/inconnues, gains d’aspiration).")
    if facteur_correctif > 1.20:
        st.warning("Facteur > 1.20 : vérifie et documente la raison (aspiration, mesures, etc.).")

    # Surface (stockée en m²)
    surface_m2_state = float(st.session_state.get("surface_m2", 150.0))
    if unit_mode.startswith("Imp"):
        surface_ft2_in = colc3.number_input("Surface de capteur (pi²)", min_value=10.0,
                                            value=float(m2_to_ft2(surface_m2_state)), step=10.0)
        surface_m2 = ft2_to_m2(surface_ft2_in)
    else:
        surface_m2 = colc3.number_input("Surface de capteur (m²)", min_value=1.0,
                                        value=float(surface_m2_state), step=1.0)
    st.session_state["surface_m2"] = float(surface_m2)

    # Pertes saisonnières
    ombrage_saison = st.slider("Ombrage – période d'utilisation (%)", 0, 90,
                               int(st.session_state.get("ombrage_saison", 10)), 1)
    st.session_state["ombrage_saison"] = int(ombrage_saison)
    atten_vent = st.slider("Atténuation des vents – période d'utilisation (%)", 0, 50,
                           int(st.session_state.get("atten_vent", 0)), 1,
                           help="Pertes supplémentaires dues au vent.")
    st.session_state["atten_vent"] = int(atten_vent)

    st.caption(f"ℹ️ {defaults['comment']}")

# ——— 2.2 Dimensionnement par débit (vise 8–10 CFM/pi²) ———
with st.expander("Débit d’air & dimensionnement (SRCC 8–10 CFM/pi²)", expanded=True):
    # Saisie du débit total
    if unit_mode.startswith("Imp"):
        qv_cfm = st.number_input("Débit volumique total (CFM)", min_value=0.0, value=float(st.session_state.get("qv_cfm", 0.0)), step=50.0)
        qv_lps = cfm_to_lps(qv_cfm)
    else:
        qv_lps = st.number_input("Débit volumique total (L/s)", min_value=0.0, value=float(st.session_state.get("qv_lps", 0.0)), step=10.0)
        qv_cfm = lps_to_cfm(qv_lps)

    # Cible configurable (par défaut 8–10 CFM/pi²)
    c1, c2 = st.columns(2)
    target_lo = c1.number_input("Cible basse (CFM/pi²)", 5.0, 20.0, 8.0, 0.5)
    target_hi = c2.number_input("Cible haute (CFM/pi²)", 5.0, 20.0, 10.0, 0.5)
    target_mid = 0.5*(target_lo + target_hi)

    # Recommandation de surface pour atteindre la cible
    surface_ft2_needed_mid = (qv_cfm / target_mid) if target_mid > 0 else 0.0
    surface_ft2_needed_lo  = (qv_cfm / target_hi) if target_hi > 0 else 0.0  # surface MIN pour ne pas dépasser la cible haute
    surface_ft2_needed_hi  = (qv_cfm / target_lo) if target_lo > 0 else 0.0  # surface MAX pour rester au-dessus de la cible basse
    surface_m2_needed_mid  = ft2_to_m2(surface_ft2_needed_mid)
    surface_m2_needed_lo   = ft2_to_m2(surface_ft2_needed_lo)
    surface_m2_needed_hi   = ft2_to_m2(surface_ft2_needed_hi)

    st.markdown(
        f"**Surface recommandée** pour viser ~{target_mid:.1f} CFM/pi² : "
        f"{surface_ft2_needed_mid:,.0f} pi² (≈ {surface_m2_needed_mid:,.1f} m²) "
        f"— plage acceptable : {surface_ft2_needed_lo:,.0f}–{surface_ft2_needed_hi:,.0f} pi² "
        f"(≈ {surface_m2_needed_lo:,.1f}–{surface_m2_needed_hi:,.1f} m²)."
    )
    if st.button("👉 Appliquer la surface recommandée (valeur médiane)"):
        st.session_state["surface_m2"] = float(surface_m2_needed_mid)
        surface_m2 = float(surface_m2_needed_mid)
        st.toast("Surface mise à jour selon la cible 8–10 CFM/pi².")

    # Option : déduire largeur/hauteur
    st.markdown("**Dimensions (option)**")
    colD1, colD2, colD3 = st.columns(3)
    mode_dim = colD1.selectbox("Mode", ["Aire seule", "Largeur×Hauteur"], index=0)
    ratio_H_over_W = colD2.number_input("Ratio H/L (si inconnu)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    incr = 0.1 if unit_mode.startswith("Métrique") else 0.5  # pas d’arrondi

    if mode_dim == "Largeur×Hauteur":
        if unit_mode.startswith("Imp"):
            largeur_ft = colD1.number_input("Largeur (ft)", min_value=1.0, value=10.0, step=incr)
            # calc hauteur à partir de l'aire
            hauteur_ft = max((m2_to_ft2(surface_m2) / max(largeur_ft, 1e-6)), incr)
            colD3.metric("Hauteur calculée", f"{hauteur_ft:,.1f} ft")
        else:
            largeur_m = colD1.number_input("Largeur (m)", min_value=0.5, value=3.0, step=incr)
            hauteur_m = max((surface_m2 / max(largeur_m, 1e-6)), incr)
            colD3.metric("Hauteur calculée", f"{hauteur_m:,.2f} m")
    else:
        # Propose largeur/hauteur selon ratio
        if unit_mode.startswith("Imp"):
            aire_ft2 = m2_to_ft2(surface_m2)
            largeur_ft = (aire_ft2 / ratio_H_over_W)**0.5
            hauteur_ft = ratio_H_over_W * largeur_ft
            colD3.metric("Proposition", f"{largeur_ft:,.1f} ft × {hauteur_ft:,.1f} ft")
        else:
            largeur_m = (surface_m2 / ratio_H_over_W)**0.5
            hauteur_m = ratio_H_over_W * largeur_m
            colD3.metric("Proposition", f"{largeur_m:,.2f} m × {hauteur_m:,.2f} m")

    # Débit surfacique obtenu avec la surface actuelle
    surface_ft2 = m2_to_ft2(surface_m2)
    eps_cfm_ft2 = qv_cfm / max(surface_ft2, 1e-9)
    eps_lps_m2  = qv_lps  / max(surface_m2, 1e-9)

    # Feedback (8–10 CFM/pi²)
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Débit surfacique", f"{eps_cfm_ft2:,.2f} CFM/pi²")
    colm2.metric("Débit surfacique", f"{eps_lps_m2:,.1f} L/s·m²")
    # Efficacité estimée via interpolation simple (courbe SRCC indicative)
    import numpy as np, matplotlib.pyplot as plt
    x = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12])    # CFM/pi²
    y = np.array([8,   18, 28, 48, 63, 74, 80, 86, 90, 91, 92, 93])  # %
    eff_est = float(np.interp(np.clip(eps_cfm_ft2, x.min(), x.max()), x, y))
    colm3.metric("Efficacité estimée", f"{eff_est:,.0f} %")

    if target_lo <= eps_cfm_ft2 <= target_hi:
        st.success(f"✅ Dans la **zone SRCC** ({target_lo:.1f}–{target_hi:.1f} CFM/pi²).")
    elif (target_lo-2) <= eps_cfm_ft2 <= (target_hi+2):
        st.warning("🟠 Proche de la zone 8–10 CFM/pi² : ajuste légèrement **surface** et/ou **débit**.")
    else:
        st.error("🔴 Hors cible : recalibre **surface** (ou ventilateur) pour approcher **8–10 CFM/pi²**.")

    # Graphique compact et interactif (point + zone cible)
    fig = plt.figure(figsize=(5, 2.6))
    ax = plt.gca()
    ax.plot(x, y)
    ax.axvspan(target_lo, target_hi, alpha=0.15)    # zone 8–10
    ax.scatter([min(max(eps_cfm_ft2, x.min()), x.max())], [np.interp(min(max(eps_cfm_ft2, x.min()), x.max()), x, y)])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Débit d'air surfacique (CFM/pi²)")
    ax.set_ylabel("Efficacité (%)")
    ax.set_title("Efficacité vs Débit surfacique (SRCC — indicatif)")
    plt.tight_layout()
    st.pyplot(fig)

    # Persistance pour les blocs suivants & export
    st.session_state["qv_lps"] = float(qv_lps)
    st.session_state["qv_cfm"] = float(qv_cfm)
    st.session_state["eps_cfm_ft2"] = float(eps_cfm_ft2)
    st.session_state["eps_lps_m2"]  = float(eps_lps_m2)

# ——— 2.3 Synthèse rapide Bloc 2 ———
st.markdown("### Synthèse Bloc 2")
colS1, colS2, colS3 = st.columns(3)
colS1.metric("Surface capteur", f"{(m2_to_ft2(surface_m2) if unit_mode.startswith('Imp') else surface_m2):,.1f} " + ("pi²" if unit_mode.startswith("Imp") else "m²"))
colS2.metric("Débit volumique", f"{(st.session_state.get('qv_cfm',0.0) if unit_mode.startswith('Imp') else st.session_state.get('qv_lps',0.0)):,.0f} " + ("CFM" if unit_mode.startswith("Imp") else "L/s"))
eps_display = st.session_state.get("eps_cfm_ft2", 0.0)
if 8.0 <= eps_display <= 10.0:
    colS3.metric("Débit surfacique", f"{eps_display:,.2f} CFM/pi² ✅")
elif 6.0 <= eps_display <= 12.0:
    colS3.metric("Débit surfacique", f"{eps_display:,.2f} CFM/pi² ⚠️")
else:
    colS3.metric("Débit surfacique", f"{eps_display:,.2f} CFM/pi² 🔴")
st.caption("🎯 Règle : dimensionner pour rester **8–10 CFM/pi²** (≈ **40–51 L/s·m²**) sur la période d'utilisation.")

# ==============================
# BLOC 3 – Coûts & Économies
# ==============================
st.header("3) Coûts & Économies")

# 3.1 Performance thermique et disponibilité
colp1, colp2, colp3 = st.columns(3)
eta0 = colp1.number_input("Rendement nominal η₀ (fraction)", value=0.65, min_value=0.1, max_value=0.9, step=0.01)
sys_derate = colp2.number_input("Pertes système (ventilateur, fuites, etc.) %", value=5.0, min_value=0.0, max_value=20.0, step=0.5)
frac_saison = colp3.slider("Part de l’irradiation utile (chauffe) %", min_value=20, max_value=100, value=70, step=5)

colp4, colp5 = st.columns(2)
avail = colp4.slider("Disponibilité opérationnelle (%)", min_value=50, max_value=100, value=95, step=1)
perte_globale = (1 - sys_derate/100.0) * (frac_saison/100.0) * (avail/100.0)
colp5.metric("Facteur de disponibilité global", f"{perte_globale*100:,.1f}%")

# Chaleur utile annuelle
if annual_kwh_m2_utile is not None:
    q_util_kwh = surface_m2 * annual_kwh_m2_utile * eta0 * (1 - sys_derate/100.0) * (frac_saison/100.0) * (avail/100.0)
    st.subheader("🔸 Chaleur utile estimée")
    st.metric("Q utile (kWh/an)", f"{q_util_kwh:,.0f}")
else:
    q_util_kwh = 0.0
    st.info("Saisir ou importer l’irradiation pour calculer la chaleur utile.")

# 3.2 Substitution énergétique & économies
energie_cible = st.selectbox("Énergie remplacée principalement", ["Gaz naturel", "Électricité", "Autre (kWh équivalent)"])
colc1, colc2, colc3 = st.columns(3)
prix_gaz_kwh = colc1.number_input("Prix gaz naturel ($/kWh PCI)", value=0.050, format="%.3f")
prix_el_kwh = colc2.number_input("Prix électricité ($/kWh)", value=0.100, format="%.3f")
rendement_chauffage = colc3.number_input("Rendement chauffage existant (%)", value=85.0, min_value=40.0, max_value=100.0, step=1.0)

kwh_final_evit = 0.0
eco_dollars = 0.0
ges_tonnes = 0.0

if q_util_kwh > 0:
    rdt = max(rendement_chauffage/100.0, 1e-6)
    if energie_cible == "Gaz naturel":
        val_kwh = prix_gaz_kwh
        ges_factor = CO2_KG_PER_KWH_NG
    elif energie_cible == "Électricité":
        val_kwh = prix_el_kwh
        ges_factor = CO2_KG_PER_KWH_QC
    else:
        colx1, colx2 = st.columns(2)
        val_kwh = colx1.number_input("Tarif ($/kWh équivalent)", value=0.070, format="%.3f")
        ges_factor = colx2.number_input("Facteur GES (kg CO₂e/kWh)", value=0.100, format="%.3f")

    kwh_final_evit = q_util_kwh / rdt
    eco_dollars = kwh_final_evit * val_kwh
    ges_tonnes = (kwh_final_evit * ges_factor) / 1000.0

    met1, met2, met3 = st.columns(3)
    met1.metric("Énergie finale évitée (kWh/an)", f"{kwh_final_evit:,.0f}")
    met2.metric("Économies annuelles ($/an)", f"{eco_dollars:,.0f}")
    met3.metric("GES évités (t CO₂e/an)", f"{ges_tonnes:,.2f}")

# 3.3 Coûts, marge & subventions
st.subheader("Coûts, marge & subventions")
colk1, colk2, colk3 = st.columns(3)
cout_mat_pi2 = colk1.number_input("Matériaux ($/pi²)", value=24.0, step=1.0)
cout_mo_pi2  = colk1.number_input("Main-d'œuvre ($/pi²)", value=12.0, step=1.0)
autres_fixes = colk2.number_input("Autres coûts fixes ($)", value=0.0, step=500.0)

marge_pct = colk2.number_input("Marge (%)", value=20.0, min_value=0.0, max_value=50.0, step=1.0)
sub_type = colk3.selectbox("Type de subvention", ["Aucune", "% du CAPEX", "$ par m² (plafonné)"])

area_m2 = float(st.session_state.get("surface_m2", 150.0))
area_ft2 = m2_to_ft2(area_m2)

capex_base = area_ft2 * (cout_mat_pi2 + cout_mo_pi2) + autres_fixes
marge = capex_base * (marge_pct/100.0)
capex_avant_sub = capex_base + marge

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

# 3.4 Indicateurs financiers
st.subheader("Indicateurs financiers")
colf1, colf2, colf3 = st.columns(3)
years = colf1.number_input("Horizon d’analyse (ans)", min_value=1, max_value=30, value=15, step=1)
discount = colf2.number_input("Taux d’actualisation (%)", value=6.0, min_value=0.0, max_value=20.0, step=0.5)
escal = colf3.number_input("Escalade prix énergie (%/an)", value=2.0, min_value=0.0, max_value=15.0, step=0.5)

npv_savings = 0.0
npv = -capex_net
spb = float("inf")

if eco_dollars > 0:
    r = discount/100.0
    g = escal/100.0
    t = np.arange(1, years+1)
    savings_nominal = eco_dollars * ((1+g)**(t-1))
    discount_factors = 1 / ((1+r)**t)
    npv_savings = float(np.sum(savings_nominal * discount_factors))
    npv = npv_savings - capex_net
    spb = capex_net / eco_dollars if eco_dollars > 0 else float("inf")

    f1, f2, f3 = st.columns(3)
    f1.metric("SPB simple (ans)", f"{spb:,.1f}" if math.isfinite(spb) else "∞")
    f2.metric("VAN des économies ($)", f"{npv_savings:,.0f}")
    f3.metric("VAN projet ($)", f"{npv:,.0f}")

    cum_disc = np.cumsum(savings_nominal*discount_factors) - capex_net
    fig3 = plt.figure(figsize=(6,3))
    plt.plot(t, cum_disc)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Années"); plt.ylabel("VAN cumulée ($)")
    plt.title("VAN cumulée – point mort"); plt.tight_layout(); st.pyplot(fig3)
else:
    st.info("Complète la substitution énergétique pour calculer VAN/SPB.")

# ==============================
# BLOC 4 – Résumé & Export
# ==============================
st.header("4) Résumé & Export")

# 4.1 Tableau de synthèse
resume = {
    "Latitude": [lat],
    "Longitude": [lon],
    "Azimut_deg": [azimuth],
    "Inclinaison_deg": [tilt],
    "Ombrage_%": [shading],
    "Vent_ref_ms": [vent_ref],
    "Surface_m2": [area_m2],
    "Irradiation_kWh_m2_y": [annual_kwh_m2 or 0],
    "Irradiation_utile_kWh_m2_y": [annual_kwh_m2_utile or 0],
    "Eta0": [eta0],
    "Derating_sys_%": [sys_derate],
    "Part_saison_%": [frac_saison],
    "Disponibilite_%": [avail],
    "Q_utile_kWh_y": [q_util_kwh],
    "E_finale_evitee_kWh_y": [kwh_final_evit],
    "Econ__$_y": [eco_dollars],
    "GES_tCO2e_y": [ges_tonnes],
    "CAPEX_base_$": [capex_base],
    "Marge_$": [marge],
    "Subvention_$": [sub_amount],
    "CAPEX_net_$": [capex_net],
    "SPB_ans": [spb if math.isfinite(spb) else None],
    "VAN_savings_$": [npv_savings],
    "VAN_projet_$": [npv],
}

df_out = pd.DataFrame(resume)
st.dataframe(df_out, use_container_width=True)

# 4.2 Export Excel
out_xlsx = BytesIO()
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    df_out.to_excel(writer, index=False, sheet_name="Résumé")
    # Optionnel : ajouter les tables sources si dispo
    try:
        clim_df.to_excel(writer, index=False, sheet_name="Climat")
    except Exception:
        pass
    try:
        if monthly_df is not None:
            monthly_df.to_excel(writer, index=False, sheet_name="Irradiation_mensuelle")
        if monthly_used is not None:
            monthly_used.to_excel(writer, index=False, sheet_name="Irradiation_utile")
    except Exception:
        pass

st.download_button(
    "📥 Télécharger le résumé Excel",
    data=out_xlsx.getvalue(),
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    file_name="mur_solaire_audit_flash_resume.xlsx",
)

# 4.3 Export PDF (simple)
try:
    from fpdf import FPDF
    out_pdf = BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Mur solaire – Audit Flash (v1.1)", ln=1)
    pdf.set_font_size(10)

    def line(txt):
        pdf.multi_cell(0, 6, txt)

    line(f"Site : {adresse}")
    line(f"Coordonnées : lat {lat:.6f}, lon {lon:.6f}")
    line(f"Orientation : Azimut {azimuth:.2f}° ({azimut_cardinal(azimuth)}), Tilt {tilt:.0f}°")
    line(f"Surface : {area_m2:,.0f} m²")
    line(f"Irradiation (plan) : {(annual_kwh_m2 or 0):,.0f} kWh/m²·an")
    line(f"Irradiation utile : {(annual_kwh_m2_utile or 0):,.0f} kWh/m²·an")
    line(f"Q utile : {q_util_kwh:,.0f} kWh/an")
    if eco_dollars > 0:
        line(f"Énergie finale évitée : {kwh_final_evit:,.0f} kWh/an")
        line(f"Économies : {eco_dollars:,.0f} $/an | GES évités : {ges_tonnes:,.2f} t CO₂e/an")
    line(f"CAPEX net : {capex_net:,.0f} $")
    if math.isfinite(spb):
        line(f"SPB : {spb:,.1f} ans | VAN projet : {npv:,.0f} $")

    pdf.output(out_pdf)
    st.download_button("🖨️ Télécharger le PDF (simple)", data=out_pdf.getvalue(), file_name="mur_solaire_audit_flash.pdf", mime="application/pdf")
except Exception:
    st.info("📄 Export PDF : installe `fpdf` pour activer (requirements.txt → fpdf).")

st.caption("⚠️ MVP pédagogique : à valider/étalonner avec RETScreen & mesures (rendements, climat, périodes, pertes spécifiques site).")












