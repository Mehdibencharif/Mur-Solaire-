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
col1, col2, colX = st.columns([1,1,1])

with col1:
    azimuth = st.number_input(
        "Azimut du mur (°)",
        value=float(st.session_state.get("azimuth", 0.0)),
        min_value=0.0, max_value=359.99, step=0.01,
        help="0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest."
    )

with col2:
    tilt = st.number_input(
        "Inclinaison (°)", value=float(st.session_state.get("tilt", 90.0)),
        min_value=0.0, max_value=90.0, step=1.0,
        help="0° = horizontal (toit), 90° = vertical (façade)."
    )

with colX:
    if st.button("🎲 Azimut aléatoire"):
        import random
        azimuth = float(random.uniform(0, 359.99))

st.session_state.update({
    "azimuth": float(azimuth),
    "tilt": float(tilt),
})

st.caption(
    f"🧭 **Azimut MUR** : {azimuth:.2f}° ({azimut_cardinal(azimuth)}) • "
    f"📐 **Inclinaison** : {tilt:.0f}° • "
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

# =========================================================
# BLOC CLIMAT — Auto (Meteostat) avec DIAGNOSTIC + Plans B
# =========================================================

DEFAULT_LAT, DEFAULT_LON = 46.813900, -71.208000
st.subheader("Climat du site – aperçu (normales 1991–2020)")

# ---- Adresse & coords depuis l'appli
adresse = st.session_state.get("adresse", "").strip()
lat = float(st.session_state.get("lat", DEFAULT_LAT))
lon = float(st.session_state.get("lon", DEFAULT_LON))

# ---- Helpers
MOIS_FR = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

def compute_degree_days(df, base_heat=18.0, base_cool=10.0, year=None):
    import calendar
    from datetime import date
    year = year or date.today().year
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    T = np.asarray(df["Temp. air (°C)"], dtype=float)
    out = df.copy()
    out["DD18 (°C·j)"] = np.round(np.maximum(0.0, base_heat - T) * days, 0)
    out["DD10 (°C·j)"] = np.round(np.maximum(0.0, T - base_cool) * days, 0)
    return out

DEFAULT_CLIMATE_SADM = {
    "Mois": MOIS_FR,
    "Temp. air (°C)": [-12.4, -11.0, -4.6, 3.3, 10.8, 16.3, 19.1, 17.2, 12.5, 6.5, 0.5, -9.1],
    "HR (%)": [69.1, 66.8, 66.1, 64.4, 64.0, 68.8, 73.6, 74.1, 75.9, 74.1, 74.1, 75.0],
    "Précip. (mm)": [68.29, 64.52, 79.27, 81.89, 96.29, 119.33, 122.19, 114.88, 102.99, 112.61, 101.26, 92.38],
    "Pression (kPa)": [100.6, 100.6, 100.5, 100.5, 100.6, 100.5, 100.4, 100.5, 100.7, 100.8, 100.7, 100.7],
    "Vent (m/s)": [4.7, 4.7, 4.7, 4.5, 4.2, 3.6, 3.1, 3.4, 3.3, 3.9, 4.3, 4.5],
}
DEFAULT_CLIMATE_SADM = compute_degree_days(pd.DataFrame(DEFAULT_CLIMATE_SADM))

# ---- Source
source_climat = st.radio(
    "Source des données climatiques",
    ["Auto (station la plus proche)", "Préréglage SADM"],
    index=0, horizontal=True
)

# ---- DIAGNOSTIC FLAGS
diag = {
    "lib_meteostat_ok": False,
    "internet_ok": False,
    "stations_found": False,
    "normals_found": False,
    "error": None,
    "station_label": None,
}

# ---- Check internet (rapide, non bloquant)
def check_internet():
    try:
        import socket
        socket.gethostbyname("www.google.com")
        s = socket.create_connection(("www.google.com", 80), 2)
        s.close()
        return True
    except Exception:
        return False

# ---- Bloc principal
df_clim, meta = None, None

if source_climat.startswith("Auto"):
    # 1) Dispo librairie
    try:
        from meteostat import Stations, Normals
        diag["lib_meteostat_ok"] = True
    except Exception as e:
        diag["error"] = f"Meteostat non installé ou import impossible: {e}"

    # 2) Internet
    if diag["lib_meteostat_ok"]:
        diag["internet_ok"] = check_internet()

    # 3) Stations + normales
    if diag["lib_meteostat_ok"] and diag["internet_ok"]:
        try:
            stns = Stations().nearby(lat, lon).fetch(3)
            if stns is not None and not stns.empty:
                diag["stations_found"] = True

                # Choix station (et distance si dispo)
                options = []
                for idx, row in stns.iterrows():
                    name = str(row.get("name", idx))
                    country = row.get("country", "")
                    dist_km = row.get("distance", np.nan)
                    if pd.notna(dist_km):
                        label = f"{idx} – {name} ({country}) • {dist_km/1000:.1f} km"
                    else:
                        label = f"{idx} – {name} ({country})"
                    options.append((label, idx))

                selected_station_label = st.selectbox(
                    "Station Meteostat (essaie-en une autre si besoin) :",
                    [o[0] for o in options],
                    index=0
                )
                chosen_id = dict(options)[selected_station_label]
                diag["station_label"] = selected_station_label

                normals = Normals(chosen_id, start=1991, end=2020).fetch()
                if normals is not None and not normals.empty:
                    diag["normals_found"] = True
                    df_clim = pd.DataFrame({
                        "Mois": MOIS_FR,
                        "Temp. air (°C)": normals.get("tavg", pd.Series([np.nan]*12)).values,
                        "HR (%)":        normals.get("rhum", pd.Series([np.nan]*12)).values,
                        "Précip. (mm)":  normals.get("prcp", pd.Series([np.nan]*12)).values,
                        "Vent (m/s)":    (normals.get("wspd", pd.Series([np.nan]*12)) / 3.6).values,
                        "Pression (kPa)":(normals.get("pres", pd.Series([np.nan]*12)) / 10.0).values
                    })
                    df_clim = compute_degree_days(df_clim)
                    meta = {"id": chosen_id, "name": stns.loc[chosen_id].get("name", chosen_id),
                            "country": stns.loc[chosen_id].get("country", ""),
                            "distance_m": stns.loc[chosen_id].get("distance", np.nan)}
        except Exception as e:
            diag["error"] = f"Echec fetch stations/normales: {e}"

# ---- Fallback si Auto KO
if df_clim is None:
    if source_climat.startswith("Auto"):
        # Affiche la raison exacte
        if not diag["lib_meteostat_ok"]:
            st.error("❌ Meteostat indisponible. Ajoute à ton requirements.txt : `meteostat`, `numpy`, `pandas`.")
        elif not diag["internet_ok"]:
            st.error("🌐 Pas d’accès Internet sortant : l’Auto ne peut pas interroger les stations.")
        elif not diag["stations_found"]:
            st.error("📡 Aucune station trouvée pour ces coordonnées.")
        elif not diag["normals_found"]:
            st.error("📊 Normales indisponibles pour la station sélectionnée.")
        if diag["error"]:
            st.caption(f"🧪 Détail technique : {diag['error']}")
        st.warning("→ Utilisation du préréglage **SADM** ci-dessous.")
    df_clim = DEFAULT_CLIMATE_SADM.copy()
    meta = None

# ---- Affichage tableau + métriques
st.dataframe(df_clim, use_container_width=True, hide_index=True)
moy_air = float(df_clim["Temp. air (°C)"].mean(skipna=True))
sum_dd18 = float(df_clim["DD18 (°C·j)"].sum(skipna=True))
sum_dd10 = float(df_clim["DD10 (°C·j)"].sum(skipna=True))

c1, c2 = st.columns(2)
c1.metric("T° air moyenne", f"{moy_air:.1f} °C")
c2.metric("DD18 / DD10", f"{sum_dd18:,.0f} / {sum_dd10:,.0f} °C·j")

# ---- Légende
if source_climat.startswith("Auto") and meta:
    dist_txt = f" • {meta['distance_m']/1000:.1f} km" if pd.notna(meta.get("distance_m", np.nan)) else ""
    st.caption(f"📡 Station : **{meta['name']}** ({meta['country']}) — id **{meta['id']}**{dist_txt}")
else:
    st.caption("📘 Préréglage **SADM** (valeurs type) — à valider/affiner avec RETScreen.")

# ---- PANNEAU DEBUG (temporaire)
with st.expander("🛠️ Debug climat (temporaire)"):
    st.write({"adresse": adresse, "lat": lat, "lon": lon})
    st.write(diag)

# =========================================================
# BLOC 2 — Charge & exploitation (style RETScreen)
# =========================================================
st.header("2) Charge & exploitation")

unit_mode = st.session_state.get("unit_mode", "Métrique (SI)")

# Conversions
FT2_PER_M2  = 10.7639
M2_PER_FT2  = 1.0 / FT2_PER_M2
CFM_PER_LPS = 2.11888
LPS_PER_CFM = 1.0 / CFM_PER_LPS
R_SI_PER_R_US = 0.1761101838   # m²·K/W par (h·ft²·°F/Btu)
R_US_PER_R_SI = 1.0 / R_SI_PER_R_US

def m2_to_ft2(x): return x * FT2_PER_M2
def ft2_to_m2(x): return x * M2_PER_FT2
def lps_to_cfm(x): return x * CFM_PER_LPS
def cfm_to_lps(x): return x * LPS_PER_LPS

# ---------------- Caractéristiques de la charge ----------------
with st.expander("Caractéristiques de la charge (réf./proposé)", expanded=True):
    colA, colB = st.columns(2)
    type_install = colA.selectbox("Type d'installation", ["Industriel", "Commercial", "Institutionnel", "Autre"], index=0)

    # Températures
    cT1, cT2, cT3, cT4 = st.columns(4)
    t_int = cT1.number_input("Température intérieure (°C)", value=float(st.session_state.get("t_int", 20.0)), step=0.5)
    t_min = cT2.number_input("T° air - minimum (°C)", value=float(st.session_state.get("t_min", -10.0)), step=0.5)
    t_max = cT3.number_input("T° air - maximum (°C)", value=float(st.session_state.get("t_max", 25.0)), step=0.5)
    strat = cT4.number_input("Stratification intérieure (°C)", value=float(st.session_state.get("strat", 0.0)), step=0.5)

    # Surface planchers + valeurs R (RSI en SI, R-us en imp)
    cS1, cS2, cS3 = st.columns(3)
    if unit_mode.startswith("Imp"):
        surf_ft2 = cS1.number_input("Surface de planchers (pi²)", min_value=0.0,
                                    value=float(st.session_state.get("surf_ft2", 20000.0)), step=500.0, format="%.0f")
        r_roof_us = cS2.number_input("Valeur R - plafond (h·ft²·°F/Btu)", min_value=0.1,
                                     value=float(st.session_state.get("r_roof_us", 20.0)), step=1.0)
        r_wall_us = cS3.number_input("Valeur R - mur (h·ft²·°F/Btu)", min_value=0.1,
                                     value=float(st.session_state.get("r_wall_us", 12.0)), step=1.0)
        surf_m2 = ft2_to_m2(surf_ft2)
        r_roof_si = r_roof_us * R_SI_PER_R_US
        r_wall_si = r_wall_us * R_SI_PER_R_US
    else:
        surf_m2 = cS1.number_input("Surface de planchers (m²)", min_value=0.0,
                                   value=float(st.session_state.get("surf_m2", 1858.0)), step=10.0, format="%.1f")
        r_roof_si = cS2.number_input("Valeur R - plafond (m²·°C/W)", min_value=0.1,
                                     value=float(st.session_state.get("r_roof_si", 3.5)), step=0.1)
        r_wall_si = cS3.number_input("Valeur R - mur (m²·°C/W)", min_value=0.1,
                                     value=float(st.session_state.get("r_wall_si", 2.1)), step=0.1)
        surf_ft2 = m2_to_ft2(surf_m2)
        r_roof_us = r_roof_si * R_US_PER_R_SI
        r_wall_us = r_wall_si * R_US_PER_R_SI

    # Débit d'air de conception (cohérent avec Bloc 3 si tu l'utilises déjà)
    if unit_mode.startswith("Imp"):
        qv_cfm = st.number_input("Débit d'air de conception (CFM)", min_value=0.0,
                                 value=float(st.session_state.get("qv_cfm", 10000.0)), step=100.0, format="%.0f")
        qv_lps = qv_cfm / CFM_PER_LPS
    else:
        qv_lps = st.number_input("Débit d'air de conception (L/s)", min_value=0.0,
                                 value=float(st.session_state.get("qv_lps", 5000.0)), step=50.0, format="%.0f")
        qv_cfm = qv_lps * CFM_PER_LPS

# ---------------- Horaire d’opération ----------------
with st.expander("Horaire d’opération", expanded=True):
    cH1, cH2, cH3, cH4 = st.columns(4)
    j_sem   = cH1.number_input("Jours/sem (semaine)", min_value=0, max_value=7, value=int(st.session_state.get("j_sem", 5)))
    h_j_sem = cH2.number_input("Heures/jour (semaine)", min_value=0.0, max_value=24.0, value=float(st.session_state.get("h_j_sem", 8.0)), step=0.5)
    j_wkd   = cH3.number_input("Jours/sem (fins de semaine)", min_value=0, max_value=2, value=int(st.session_state.get("j_wkd", 0)))
    h_j_wkd = cH4.number_input("Heures/jour (fins de semaine)", min_value=0.0, max_value=24.0, value=float(st.session_state.get("h_j_wkd", 0.0)), step=0.5)

    heures_sem = j_sem * h_j_sem + j_wkd * h_j_wkd
    st.caption(f"⏱️ **Heures d'opération / semaine** : **{heures_sem:.1f} h/sem** (≈ {heures_sem*52:.0f} h/an)")

# ---------------- Portion d'utilisation mensuelle + Rayonnement ----------------
with st.expander("Portion d’utilisation mensuelle & rayonnement", expanded=True):
    # Table d’usage (réf/proposé)
    mois_labels = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    usage_df = pd.DataFrame({
        "Mois": mois_labels,
        "Utilisation % (réf.)":    [100]*12,
        "Utilisation % (proposé)": [100]*12,
    })
    col_cfg = {
        "Utilisation % (réf.)": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%d"),
        "Utilisation % (proposé)": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%d"),
    }
    usage_df = st.data_editor(usage_df, hide_index=True, use_container_width=True, column_config=col_cfg)
    for c in ["Utilisation % (réf.)","Utilisation % (proposé)"]:
        usage_df[c] = pd.to_numeric(usage_df[c], errors="coerce").fillna(0).clip(0, 100)

    # Rayonnement quotidien (kWh/m²/j) — depuis le climat du Bloc 1 s'il existe
    clim_df = st.session_state.get("climat_mensuel_df", None)
    if (clim_df is not None) and ("Rayon. horiz. (kWh/m²/j)" in clim_df.columns):
        df_ray = pd.DataFrame({"Mois": mois_labels})
        df_ray["Rayonnement quotidien - horizontal (kWh/m²/j)"] = np.asarray(clim_df["Rayon. horiz. (kWh/m²/j)"], dtype=float)

        # Incliné (si tu n'as pas mieux ici, on duplique à titre d'aperçu ; le vrai “sur plan du mur” sera en Bloc 3)
        df_ray["Rayonnement quotidien - incliné (kWh/m²/j)"] = df_ray["Rayonnement quotidien - horizontal (kWh/m²/j)"]

        # Calcul moyenne annuelle quotidienne + annuel (MWh/m²)
        import calendar
        days = np.array([calendar.monthrange(date.today().year, m)[1] for m in range(1,13)])
        daily_avg = (df_ray["Rayonnement quotidien - horizontal (kWh/m²/j)"] * days).sum() / days.sum()
        annual_MWh = daily_avg * 365.0 / 1000.0

        st.dataframe(df_ray, hide_index=True, use_container_width=True)
        st.caption(f"📊 **Rayonnement quotidien moyen (horizontal)** : {daily_avg:.2f} kWh/m²/j • **Annuel** ≈ {annual_MWh:.2f} MWh/m²")

    else:
        st.info("Pas de colonne **Rayon. horiz. (kWh/m²/j)** dans le climat du Bloc 1. Tu pourras fournir le mensuel RETScreen au Bloc 3.")

# ---------------- Stockage pour la suite ----------------
st.session_state["load_params"] = {
    "type_install": type_install,
    "t_int_C": float(t_int), "t_min_C": float(t_min), "t_max_C": float(t_max), "strat_C": float(strat),
    "surf_m2": float(surf_m2), "surf_ft2": float(surf_ft2),
    "R_roof_SI": float(r_roof_si), "R_wall_SI": float(r_wall_si),
    "R_roof_US": float(r_roof_us), "R_wall_US": float(r_wall_us),
    "qv_lps": float(qv_lps), "qv_cfm": float(qv_cfm),
    "j_sem": int(j_sem), "h_j_sem": float(h_j_sem), "j_wkd": int(j_wkd), "h_j_wkd": float(h_j_wkd),
    "heures_sem": float(heures_sem), "heures_an": float(heures_sem*52.0),
}
st.session_state["usage_mensuel_charge"] = usage_df

# ==============================
# BLOC 3 – Paramètres du capteur solaire à air
# ==============================
st.header("3) Paramètres du capteur solaire à air")

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
    absorptivite = colc1.number_input("Absorptivité du capteur",
                                      min_value=0.80, max_value=0.99,
                                      value=float(defaults["absorptivite"]), step=0.01)
    couleur = colc1.selectbox("Couleur/finition", ["Noir", "Anthracite", "Autre"], index=0)

    facteur_correctif = colc2.number_input("Facteur correctif global (adim.)",
                                           min_value=0.50, max_value=2.00,
                                           value=float(defaults["facteur_correctif"]), step=0.01,
                                           help="Calage global (ombrage résiduel, pertes/inconnues, gains d’aspiration).")
    if facteur_correctif > 1.20:
        st.warning("Facteur > 1.20 : vérifie et documente la raison (aspiration, mesures, etc.).")

    # Surface (stockée en m²) — clamp des valeurs par défaut pour éviter StreamlitValueBelowMinError
    surface_m2_state = float(st.session_state.get("surface_m2", 150.0))
    if unit_mode.startswith("Imp"):
        surface_ft2_default = max(float(m2_to_ft2(surface_m2_state)), 1.0)  # ≥ min
        surface_ft2_in = colc3.number_input(
            "Surface de capteur (pi²)",
            min_value=1.0,                          # ↓ min tolérant
            value=surface_ft2_default,              # ↓ valeur clampée
            step=50.0, format="%.0f",
            help="Surface nette exposée (pi²)."
        )
        surface_m2 = ft2_to_m2(surface_ft2_in)
    else:
        surface_m2_default = max(surface_m2_state, 0.1)          # ≥ min
        surface_m2 = colc3.number_input(
            "Surface de capteur (m²)",
            min_value=0.1,                         # ↓ min tolérant
            value=surface_m2_default,              # ↓ valeur clampée
            step=1.0,
            help="Surface nette exposée (m²)."
        )
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
    # Saisie du débit total (grandes valeurs OK)
    if unit_mode.startswith("Imp"):
        qv_cfm = st.number_input(
            "Débit volumique total (CFM)",
            min_value=0.0, value=float(st.session_state.get("qv_cfm", 10000.0)),
            step=100.0, format="%.0f"
        )
        qv_lps = cfm_to_lps(qv_cfm)
    else:
        qv_lps = st.number_input(
            "Débit volumique total (L/s)",
            min_value=0.0, value=float(st.session_state.get("qv_lps", 5000.0)),
            step=50.0, format="%.0f"
        )
        qv_cfm = lps_to_cfm(qv_lps)

    # Cible configurable (par défaut 8–10 CFM/pi²)
    c1, c2 = st.columns(2)
    target_lo = c1.number_input("Cible basse (CFM/pi²)", 5.0, 20.0, 8.0, 0.5)
    target_hi = c2.number_input("Cible haute (CFM/pi²)", 5.0, 20.0, 10.0, 0.5)
    target_mid = 0.5 * (target_lo + target_hi)

    # Recommandation de surface pour atteindre la cible
    surface_ft2_needed_mid = (qv_cfm / target_mid) if target_mid > 0 else 0.0
    surface_ft2_needed_lo  = (qv_cfm / target_hi)  if target_hi  > 0 else 0.0
    surface_ft2_needed_hi  = (qv_cfm / target_lo)  if target_lo  > 0 else 0.0
    surface_m2_needed_mid  = ft2_to_m2(surface_ft2_needed_mid)
    surface_m2_needed_lo   = ft2_to_m2(surface_ft2_needed_lo)
    surface_m2_needed_hi   = ft2_to_m2(surface_ft2_needed_hi)

    st.markdown(
        f"**Surface recommandée** pour viser ~{target_mid:.1f} CFM/pi² : "
        f"{surface_ft2_needed_mid:,.0f} pi² (≈ {surface_m2_needed_mid:,.1f} m²) — "
        f"plage acceptable : {surface_ft2_needed_lo:,.0f}–{surface_ft2_needed_hi:,.0f} pi² "
        f"(≈ {surface_m2_needed_lo:,.1f}–{surface_m2_needed_hi:,.1f} m²)."
    )
    if st.button("👉 Appliquer la surface recommandée (valeur médiane)"):
        st.session_state["surface_m2"] = float(surface_m2_needed_mid)
        surface_m2 = float(surface_m2_needed_mid)
        st.toast("Surface mise à jour selon la cible 8–10 CFM/pi².")

    # Dimensions (option)
    st.markdown("**Dimensions (option)**")
    colD1, colD2, colD3 = st.columns(3)
    mode_dim = colD1.selectbox("Mode", ["Aire seule", "Largeur×Hauteur"], index=0)
    ratio_H_over_W = colD2.number_input("Ratio H/L (si inconnu)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    incr = 0.1 if unit_mode.startswith("Métrique") else 0.5

    if mode_dim == "Largeur×Hauteur":
        if unit_mode.startswith("Imp"):
            largeur_ft = colD1.number_input("Largeur (ft)", min_value=1.0, value=10.0, step=incr)
            hauteur_ft = max((m2_to_ft2(surface_m2) / max(largeur_ft, 1e-6)), incr)
            colD3.metric("Hauteur calculée", f"{hauteur_ft:,.1f} ft")
        else:
            largeur_m = colD1.number_input("Largeur (m)", min_value=0.5, value=3.0, step=incr)
            hauteur_m = max((surface_m2 / max(largeur_m, 1e-6)), incr)
            colD3.metric("Hauteur calculée", f"{hauteur_m:,.2f} m")
    else:
        if unit_mode.startswith("Imp"):
            aire_ft2 = m2_to_ft2(surface_m2)
            largeur_ft = (aire_ft2 / ratio_H_over_W) ** 0.5
            hauteur_ft = ratio_H_over_W * largeur_ft
            colD3.metric("Proposition", f"{largeur_ft:,.1f} ft × {hauteur_ft:,.1f} ft")
        else:
            largeur_m = (surface_m2 / ratio_H_over_W) ** 0.5
            hauteur_m = ratio_H_over_W * largeur_m
            colD3.metric("Proposition", f"{largeur_m:,.2f} m × {hauteur_m:,.2f} m")

    # Débit surfacique obtenu avec la surface actuelle
    surface_ft2 = m2_to_ft2(surface_m2)
    eps_cfm_ft2 = qv_cfm / max(surface_ft2, 1e-9)
    eps_lps_m2  = qv_lps  / max(surface_m2, 1e-9)

    # Feedback (8–10 CFM/pi²) + efficacité estimée
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Débit surfacique", f"{eps_cfm_ft2:,.2f} CFM/pi²")
    colm2.metric("Débit surfacique", f"{eps_lps_m2:,.1f} L/s·m²")

    import numpy as np
    import matplotlib.pyplot as plt
    x = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12])       # CFM/pi²
    y = np.array([8,   18, 28, 48, 63, 74, 80, 86, 90, 91, 92, 93])  # %
    eff_est = float(np.interp(np.clip(eps_cfm_ft2, x.min(), x.max()), x, y))
    colm3.metric("Efficacité estimée", f"{eff_est:,.0f} %")

    if target_lo <= eps_cfm_ft2 <= target_hi:
        st.success(f"✅ Dans la **zone SRCC** ({target_lo:.1f}–{target_hi:.1f} CFM/pi²).")
    elif (target_lo - 2) <= eps_cfm_ft2 <= (target_hi + 2):
        st.warning("🟠 Proche de la zone 8–10 CFM/pi² : ajuste légèrement **surface** et/ou **débit**.")
    else:
        st.error("🔴 Hors cible : recalibre **surface** (ou ventilateur) pour approcher **8–10 CFM/pi²**.")

    # Graphique compact et fonctionnel
    fig = plt.figure(figsize=(4.6, 2.2))
    ax = plt.gca()
    ax.plot(x, y)
    ax.axvspan(target_lo, target_hi, alpha=0.15)      # zone 8–10
    xpt = min(max(eps_cfm_ft2, x.min()), x.max())
    ypt = float(np.interp(xpt, x, y))
    ax.scatter([xpt], [ypt])
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
colS1.metric(
    "Surface capteur",
    f"{(m2_to_ft2(surface_m2) if unit_mode.startswith('Imp') else surface_m2):,.1f} " +
    ("pi²" if unit_mode.startswith("Imp") else "m²")
)
colS2.metric(
    "Débit volumique",
    f"{(st.session_state.get('qv_cfm',0.0) if unit_mode.startswith('Imp') else st.session_state.get('qv_lps',0.0)):,.0f} " +
    ("CFM" if unit_mode.startswith("Imp") else "L/s")
)
eps_display = st.session_state.get("eps_cfm_ft2", 0.0)
if 8.0 <= eps_display <= 10.0:
    colS3.metric("Débit surfacique", f"{eps_display:,.2f} CFM/pi² ✅")
elif 6.0 <= eps_display <= 12.0:
    colS3.metric("Débit surfacique", f"{eps_display:,.2f} CFM/pi² ⚠️")
else:
    colS3.metric("Débit surfacique", f"{eps_display:,.2f} CFM/pi² 🔴")
st.caption("🎯 Règle : dimensionner pour rester **8–10 CFM/pi²** (≈ **40–51 L/s·m²**) sur la période d'utilisation.")

# =========================
# SOMMAIRE (style RETScreen)
# =========================
st.subheader("Sommaire énergétique – ventilation")

# Récup des entrées utiles
surface_m2  = float(st.session_state.get("surface_m2", 0.0))
qv_lps      = float(st.session_state.get("qv_lps", 0.0))
qv_cfm      = float(st.session_state.get("qv_cfm", 0.0))
heures_an   = float(st.session_state.get("load_params", {}).get("heures_an", 2000.0))  # Bloc 2
annual_kwh_m2_utile = locals().get("annual_kwh_m2_utile", None) or float(st.session_state.get("annual_kwh_m2_utile", 0.0))

# 1) Électricité supplémentaire – ventilation
with st.expander("Électricité supplémentaire – ventilation", expanded=True):
    mode_fan = st.radio("Méthode de calcul", ["SFP (kW/(m³/s))", "ΔP & η"], horizontal=True, key="fan_method")
    qv_m3s = qv_lps / 1000.0  # 1 m³/s = 1000 L/s

    if mode_fan.startswith("SFP"):
        sfp = st.number_input("SFP (kW/(m³/s))", min_value=0.2, max_value=6.0, value=2.5, step=0.1,
                              help="Puissance spécifique des ventilateurs.")
        p_fan_kW = sfp * qv_m3s
    else:
        deltaP = st.number_input("Perte de charge totale ΔP (Pa)", min_value=0.0, value=300.0, step=10.0)
        eta_fan = st.number_input("Rendement ventilateur (%)", min_value=10.0, max_value=85.0, value=60.0, step=1.0)/100.0
        p_fan_kW = (qv_m3s * deltaP) / max(eta_fan, 1e-3) / 1000.0   # P[W] = Q[m³/s]*ΔP[Pa] ⇒ kW
    elec_kWh = p_fan_kW * heures_an
    st.metric("Électricité supplémentaire (kWh/an)", f"{elec_kWh:,.0f}")

# 2) Chaleur fournie (à partir de l’irradiation utile du mur)
with st.expander("Chaleur fournie par le capteur", expanded=True):
    if annual_kwh_m2_utile and annual_kwh_m2_utile > 0:
        eta0 = st.number_input("Rendement thermique global η", min_value=0.10, max_value=0.95, value=0.70, step=0.01)
        derate_pct = st.number_input("Pertes système (derating) %", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        part_saison = st.slider("Part de l’irradiation utile en période **chauffage** (%)", 20, 100, 70, 5)
        dispo = st.slider("Disponibilité système (%)", 70, 100, 95, 1)

        q_fournie_kWh = annual_kwh_m2_utile * surface_m2 * eta0 * (1 - derate_pct/100.0) \
                        * (part_saison/100.0) * (dispo/100.0)
        st.metric("Chaleur fournie (kWh/an)", f"{q_fournie_kWh:,.0f}")
    else:
        q_fournie_kWh = 0.0
        st.info("Renseigne l’irradiation **utile** du mur (kWh/m²·an) au-dessus pour calculer la chaleur fournie.")

# 3) Récupération des pertes de chaleur du bâtiment
with st.expander("Récupération des pertes de chaleur du bâtiment", expanded=False):
    rec_mode = st.selectbox("Méthode", ["Aucune", "Fraction de la chaleur fournie", "Par paroi (ΔU)"], index=0)
    if rec_mode == "Fraction de la chaleur fournie":
        rec_pct = st.slider("Fraction (%)", 0, 30, 0, 1, help="Usuel 0–10 % si pertinent.")
        recup_kWh = q_fournie_kWh * rec_pct/100.0
    elif rec_mode == "Par paroi (ΔU)":
        A_wall = st.number_input("Aire de mur couverte (m²)", min_value=0.0, value=float(surface_m2), step=1.0)
        U_avant = st.number_input("U mur existant (W/m²·K)", min_value=0.05, value=0.50, step=0.05)
        U_apres = st.number_input("U après mur solaire (W/m²·K)", min_value=0.01, value=0.35, step=0.05,
                                  help="Effet « pare-soleil/cavite » approximatif.")
        dT_moy = st.number_input("ΔT chauffage moyen (K)", min_value=1.0, value=20.0, step=1.0)
        heures_chauffe = st.number_input("Heures en période de chauffage (h/an)", min_value=0.0, value=float(heures_an), step=50.0)
        recup_kWh = max(U_avant - U_apres, 0.0) * A_wall * dT_moy * heures_chauffe / 1000.0
    else:
        recup_kWh = 0.0
    st.metric("Récupération de pertes (kWh/an)", f"{recup_kWh:,.0f}")

# 4) Énergie économisée par déstratification
with st.expander("Énergie économisée par déstratification", expanded=False):
    destrat_pct = st.slider("Économie (% de la chaleur fournie)", 0, 20, 0, 1,
                            help="Grand volume industriel : typiquement 0–10 %.")
    destrat_kWh = q_fournie_kWh * destrat_pct/100.0
    st.metric("Énergie économisée par déstratification (kWh/an)", f"{destrat_kWh:,.0f}")

# 5) Bilan & persistance
bilan_net_kWh = q_fournie_kWh + recup_kWh + destrat_kWh - elec_kWh

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Élec ventilos", f"{elec_kWh:,.0f} kWh/an")
s2.metric("Chaleur fournie", f"{q_fournie_kWh:,.0f} kWh/an")
s3.metric("Récup. pertes", f"{recup_kWh:,.0f} kWh/an")
s4.metric("Déstrat.", f"{destrat_kWh:,.0f} kWh/an")
s5.metric("Bilan **net**", f"{bilan_net_kWh:,.0f} kWh/an")

# Sauvegarde pour export (Bloc 5)
st.session_state["sommaire_ener"] = {
    "elec_vent_kWh": float(elec_kWh),
    "chaleur_fournie_kWh": float(q_fournie_kWh),
    "recup_pertes_kWh": float(recup_kWh),
    "destrat_kWh": float(destrat_kWh),
    "bilan_net_kWh": float(bilan_net_kWh),
    "heures_an": float(heures_an),
}
# (Optionnel) garder la valeur locale pour d'autres blocs
st.session_state["annual_kwh_m2_utile"] = float(annual_kwh_m2_utile or 0.0)


# =========================================================
# BLOC 4 — Coûts, marge & subventions (règle 75/25)
# =========================================================
st.header("4) Coûts, marge & subventions")

# Aire capteur (stockée en m² dans les blocs précédents)
FT2_PER_M2 = 10.7639
surface_m2 = float(st.session_state.get("surface_m2", 0.0))
surface_ft2 = surface_m2 * FT2_PER_M2

# ---------- Paramètres de coûts ----------
colC1, colC2, colC3, colC4 = st.columns(4)
c_mat = colC1.number_input("Matériaux ($/pi²)", min_value=0.0, value=22.0, step=1.0)
c_inst = colC2.number_input("Installation ($/pi²)", min_value=0.0, value=12.0, step=1.0)
c_mon  = colC3.number_input("Monitoring ($)", min_value=0.0, value=5000.0, step=500.0)
marge_pct = colC4.slider("Marge entreprise (%)", min_value=0, max_value=100, value=20, step=1)

# Sous-total & marge
cout_mat = surface_ft2 * c_mat
cout_inst = surface_ft2 * c_inst
sous_total = cout_mat + cout_inst + c_mon
marge = sous_total * (marge_pct / 100.0)
capex_avant_sub = sous_total + marge

# ---------- Subventions ----------
st.subheader("Subventions (Ministère + Énergir/Écoperformance)")
sub_mode = st.radio("Mode de saisie", ["% du CAPEX", "Montants fixes ($)"], horizontal=True)

if sub_mode == "% du CAPEX":
    cols = st.columns(2)
    pct_ministere = cols[0].number_input("Ministère (% du CAPEX)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    pct_energir   = cols[1].number_input("Énergir/Écoperformance (% du CAPEX)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    sub_brut = capex_avant_sub * (pct_ministere + pct_energir) / 100.0
    details_sub = f"{pct_ministere:.0f}% + {pct_energir:.0f}%"
else:
    cols = st.columns(2)
    amt_ministere = cols[0].number_input("Ministère ($)", min_value=0.0, value=0.0, step=1000.0, format="%.0f")
    amt_energir   = cols[1].number_input("Énergir/Écoperformance ($)", min_value=0.0, value=0.0, step=1000.0, format="%.0f")
    sub_brut = amt_ministere + amt_energir
    details_sub = f"{amt_ministere:,.0f}$ + {amt_energir:,.0f}$"

# Règle 75/25 : subventions ≤ 25 % du CAPEX (après marge)
plafond_sub = 0.25 * capex_avant_sub
sub_appliquee = min(sub_brut, plafond_sub)
plafonnee = sub_brut > plafond_sub

# ---------- Totaux ----------
invest_net = max(capex_avant_sub - sub_appliquee, 0.0)
solde_pct = (invest_net / capex_avant_sub * 100.0) if capex_avant_sub > 0 else 100.0
sub_pct_effectif = (sub_appliquee / capex_avant_sub * 100.0) if capex_avant_sub > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("CAPEX (base)", f"{sous_total:,.0f} $")
k2.metric("Marge", f"{marge:,.0f} $")
k3.metric("Subventions appliquées", f"{sub_appliquee:,.0f} $")
k4.metric("Investissement net", f"{invest_net:,.0f} $")

if plafonnee:
    st.warning(f"Les subventions saisies ({details_sub}) dépassent **25 %** du CAPEX ({capex_avant_sub:,.0f} $). "
               f"Un **plafond de 25 %** est appliqué automatiquement (subvention effective **{sub_pct_effectif:.1f} %**).")
else:
    st.info(f"Subventions effectives : **{sub_pct_effectif:.1f} %** du CAPEX • Solde à payer : **{solde_pct:.1f} %** (règle 75/25 respectée)")

# ---------- Persistance pour export ----------
st.session_state["couts_finance"] = {
    "surface_ft2": surface_ft2,
    "cout_mat_unit": c_mat, "cout_inst_unit": c_inst, "monitoring": c_mon,
    "cout_mat_total": cout_mat, "cout_inst_total": cout_inst,
    "marge_pct": marge_pct, "marge": marge,
    "capex_avant_sub": capex_avant_sub,
    "sub_mode": sub_mode, "sub_details_saisie": details_sub,
    "sub_brut": sub_brut, "sub_plafond_25pct": plafond_sub,
    "sub_appliquee": sub_appliquee,
    "invest_net": invest_net, "solde_pct": solde_pct, "sub_pct_effectif": sub_pct_effectif,
}

# =========================================================
# BLOC 5 — Résumé & Export (Excel + PDF)
# =========================================================
st.header("5) Résumé & export")

from io import BytesIO
from datetime import datetime

# --- Conversions & récup état ---
FT2_PER_M2 = 10.7639
unit_mode  = st.session_state.get("unit_mode", "Métrique (SI)")

# Site & orientation
adresse   = st.session_state.get("adresse", "")
lat       = float(st.session_state.get("lat", 0.0))
lon       = float(st.session_state.get("lon", 0.0))
azimuth   = float(st.session_state.get("azimuth", 0.0))
tilt      = float(st.session_state.get("tilt", 90.0))
shading   = int(st.session_state.get("shading", 0))
wind_ref  = float(st.session_state.get("wind_ref", 0.0))
clim_meta = st.session_state.get("climat_meta", {})
clim_df   = st.session_state.get("climat_mensuel_df", None)

# Charge (Bloc 2)
load_params = st.session_state.get("load_params", {})
usage_mensuel_charge = st.session_state.get("usage_mensuel_charge", None)

# Capteur & dimensionnement (Bloc 3)
surface_m2  = float(st.session_state.get("surface_m2", 0.0))
surface_ft2 = surface_m2 * FT2_PER_M2
qv_cfm      = float(st.session_state.get("qv_cfm", 0.0))
qv_lps      = float(st.session_state.get("qv_lps", 0.0))
eps_cfm_ft2 = float(st.session_state.get("eps_cfm_ft2", 0.0))
eps_lps_m2  = float(st.session_state.get("eps_lps_m2", 0.0))

# Coûts & subventions (Bloc 4)
fin = st.session_state.get("couts_finance", {})

# =================== APERÇU SYNTHÈSE ===================
c1, c2, c3 = st.columns(3)
# Surface
if unit_mode.startswith("Imp"):
    c1.metric("Surface capteur", f"{surface_ft2:,.0f} pi²")
else:
    c1.metric("Surface capteur", f"{surface_m2:,.1f} m²")
# Débit total
if unit_mode.startswith("Imp"):
    c2.metric("Débit volumique", f"{qv_cfm:,.0f} CFM")
else:
    c2.metric("Débit volumique", f"{qv_lps:,.0f} L/s")
# Débit surfacique
badge = "✅" if 8.0 <= eps_cfm_ft2 <= 10.0 else ("⚠️" if 6.0 <= eps_cfm_ft2 <= 12.0 else "🔴")
c3.metric("Débit surfacique", f"{eps_cfm_ft2:,.2f} CFM/pi² {badge}")

# Coûts *si disponibles*
if fin:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CAPEX (base)", f"{fin.get('cout_mat_total',0)+fin.get('cout_inst_total',0)+fin.get('monitoring',0):,.0f} $")
    k2.metric("Marge", f"{fin.get('marge',0):,.0f} $")
    k3.metric("Subventions", f"{fin.get('sub_appliquee',0):,.0f} $")
    k4.metric("Investissement net", f"{fin.get('invest_net',0):,.0f} $")

st.divider()

# =================== EXPORT EXCEL ===================
st.subheader("Export Excel")
resume_rows = {
    "Horodatage": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "Unités": unit_mode,
    # Site
    "Adresse": adresse, "Latitude": lat, "Longitude": lon,
    "Azimut_deg": azimuth, "Inclinaison_deg": tilt,
    "Ombrage_%": shading, "Vent_ref_m_s": wind_ref,
    # Capteur & débit
    "Surface_m2": surface_m2, "Surface_ft2": surface_ft2,
    "Qv_CFM": qv_cfm, "Qv_Lps": qv_lps,
    "EPS_CFM_pi2": eps_cfm_ft2, "EPS_Lps_m2": eps_lps_m2,
    # Finance (peut être 0)
    "CAPEX_base_$": (fin.get("cout_mat_total",0)+fin.get("cout_inst_total",0)+fin.get("monitoring",0)),
    "Marge_$": fin.get("marge",0),
    "CAPEX_avant_sub_$": fin.get("capex_avant_sub",0),
    "Subventions_$": fin.get("sub_appliquee",0),
    "Investissement_net_$": fin.get("invest_net",0),
    "Subvention_effective_%": fin.get("sub_pct_effectif",0),
}

out_xlsx = BytesIO()
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    # Feuille 1 — Résumé
    pd.DataFrame([resume_rows]).to_excel(writer, index=False, sheet_name="Résumé")

    # Feuille 2 — Site & Climat (métadonnées)
    if clim_meta:
        pd.DataFrame([clim_meta]).to_excel(writer, index=False, sheet_name="Site_Climat_meta")

    # Feuille 3 — Climat mensuel
    if clim_df is not None and isinstance(clim_df, pd.DataFrame):
        clim_df.to_excel(writer, index=False, sheet_name="Climat_mensuel")

    # Feuille 4 — Charge
    if load_params:
        pd.DataFrame([load_params]).to_excel(writer, index=False, sheet_name="Charge")

    # Feuille 5 — Usage mensuel (charge)
    if usage_mensuel_charge is not None and isinstance(usage_mensuel_charge, pd.DataFrame):
        usage_mensuel_charge.to_excel(writer, index=False, sheet_name="Usage_mensuel")

    # Feuille 6 — Coûts & finance
    if fin:
        pd.DataFrame([fin]).to_excel(writer, index=False, sheet_name="Couts_finance")

st.download_button(
    "📥 Télécharger l’Excel (Résumé complet)",
    data=out_xlsx.getvalue(),
    file_name="mur_solaire_resume.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

# =================== EXPORT PDF (simple) ===================
st.subheader("Export PDF (synthèse)")
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    pdf_path = "/mnt/data/mur_solaire_resume.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4

    # petit helper qui renvoie le nouveau y (pas de nonlocal)
    def draw_line(cvs, y, txt, size=10, dy=0.6*cm, bold=False):
        cvs.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        cvs.drawString(2*cm, y, str(txt))
        return y - dy

    y = H - 2*cm
    y = draw_line(c, y, "Résumé – Mur solaire", size=14, dy=0.9*cm, bold=True)
    y = draw_line(c, y, f"Date : {resume_rows['Horodatage']}", size=10)

    y = draw_line(c, y, "— Site —", size=12, bold=True)
    y = draw_line(c, y, f"Adresse : {adresse}")
    y = draw_line(c, y, f"Coordonnées : {lat:.5f}, {lon:.5f}")
    y = draw_line(c, y, f"Orientation : azimut {azimuth:.1f}°, inclinaison {tilt:.0f}°")
    y = draw_line(c, y, f"Ombrage : {shading} %  |  Vent réf. : {wind_ref:.1f} m/s")

    y = draw_line(c, y, "— Capteur & Débit —", size=12, bold=True)
    y = draw_line(c, y, f"Surface : {surface_ft2:,.0f} pi² (≈ {surface_m2:,.1f} m²)")
    y = draw_line(c, y, f"Débit : {qv_cfm:,.0f} CFM (≈ {qv_lps:,.0f} L/s)")
    y = draw_line(c, y, f"Débit surfacique : {eps_cfm_ft2:,.2f} CFM/pi²")

    if fin:
        y = draw_line(c, y, "— Coûts & Subventions —", size=12, bold=True)
        capex_base = fin.get("cout_mat_total",0)+fin.get("cout_inst_total",0)+fin.get("monitoring",0)
        y = draw_line(c, y, f"CAPEX (base) : {capex_base:,.0f} $")
        y = draw_line(c, y, f"Marge : {fin.get('marge',0):,.0f} $  |  CAPEX avant subventions : {fin.get('capex_avant_sub',0):,.0f} $")
        y = draw_line(c, y, f"Subventions appliquées : {fin.get('sub_appliquee',0):,.0f} $  (≈ {fin.get('sub_pct_effectif',0):.1f} %)")
        y = draw_line(c, y, f"Investissement net : {fin.get('invest_net',0):,.0f} $")

    c.showPage()
    c.save()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        "📄 Télécharger le PDF (synthèse)",
        data=pdf_bytes,
        file_name="mur_solaire_resume.pdf",
        mime="application/pdf",
        use_container_width=True
    )
except Exception:
    st.info("Export PDF indisponible (bibliothèque **reportlab** manquante). L’export **Excel** reste complet.")






















