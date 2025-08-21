import os
import calendar
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
from urllib.parse import quote_plus
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Mur solaire – Audit Flash", layout="wide")
st.title("Audit Flash – Mur solaire")
st.caption("V1.0 – Prototype : localisation + azimut + climat prérempli (type RETScreen).")

# =========================================================
# SECTION 1 — LOCALISATION & ORIENTATION (AZIMUT + CARTE)
# =========================================================
st.header("1) Localisation & Orientation")

# Adresse + liens
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

# Géocodage
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

st.session_state["lat"] = float(lat)
st.session_state["lon"] = float(lon)

if not coords and adresse.strip():
    st.warning("Géocodage indisponible ou infructueux. Coordonnées par défaut affichées — ajuste-les au besoin.")

with st.expander("Comment mesurer/valider l’azimut ?"):
    st.write(
        "- L’**azimut** est mesuré **depuis le Nord**, en degrés et **sens horaire**.\n"
        "- **0°** = Nord, **90°** = Est, **180°** = Sud, **270°** = Ouest.\n"
        "- ⚠️ Valeur **0–359.99°** (jamais négative)."
    )

# ====== Orientation du mur par 2 points (RECOMMANDÉ) ======
with st.expander("Définir l’azimut du MUR par 2 points (Google Maps/Earth)", expanded=False):
    colp1, colp2 = st.columns(2)
    with colp1:
        lat_A = st.text_input("Lat A", "")
        lon_A = st.text_input("Lon A", "")
    with colp2:
        lat_B = st.text_input("Lat B", "")
        lon_B = st.text_input("Lon B", "")

    def _to_float(x):
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return None

    def bearing_from_points(lat1, lon1, lat2, lon2):
        import math as m
        φ1, φ2 = m.radians(lat1), m.radians(lat2)
        Δλ = m.radians(lon2 - lon1)
        y = m.sin(Δλ) * m.cos(φ2)
        x = m.cos(φ1)*m.sin(φ2) - m.sin(φ1)*m.cos(φ2)*m.cos(Δλ)
        θ = m.degrees(m.atan2(y, x))
        return (θ + 360.0) % 360.0  # 0–360°, depuis le Nord (horaire)

    if all(v != "" for v in [lat_A, lon_A, lat_B, lon_B]):
        la, loa, lb, lob = map(_to_float, [lat_A, lon_A, lat_B, lon_B])
        if None not in (la, loa, lb, lob):
            az_wall = bearing_from_points(la, loa, lb, lob)
            st.success(f"Azimut du **mur** = {az_wall:.2f}° (0=N, 90=E, 180=S, 270=O)")
            if st.button("👍 Utiliser cet azimut pour le mur"):
                st.session_state["azimuth"] = float(az_wall)
                st.toast("Azimut du mur mis à jour.")
        else:
            st.warning("Coordonnées invalides. Utilise des nombres (ex. 46.8139 et -71.2080).")

# ====== Info (facultatif) : azimut du SOLEIL — ne PAS confondre avec l’azimut du mur ======
st.subheader("Info : azimut du **soleil** (ne pas confondre avec l’orientation du mur)")

import datetime as dt
# Choix du fuseau
try:
    import pytz
    tz_default = "America/Toronto"
    tz = st.selectbox("Fuseau horaire", options=[tz_default, "UTC"], index=0)
    tzinfo = pytz.timezone(tz)
except Exception:
    tzinfo = None
    tz = "UTC"
    st.info("pytz non disponible — utilisation UTC.")

# Paramètres temporels
today_local = dt.datetime.now().date()
col_d, col_t, col_noon = st.columns([1,1,1])
with col_d:
    date_sel = st.date_input("Date", value=today_local, help="Date d'évaluation de la position du soleil.")
with col_t:
    time_sel = st.time_input("Heure locale", value=dt.time(12, 0), help="Heure locale d'évaluation.")
with col_noon:
    use_solar_noon = st.checkbox("Afficher au midi solaire", value=True,
                                 help="Hauteur maximale du soleil — azimut ≈ 180° au Québec.")

def compute_solar_noon(lat, lon, date_obj, tzinfo):
    # Astral v2+
    try:
        from astral import Observer
        from astral.sun import solar_noon
        tzname = tzinfo.zone if tzinfo else "UTC"
        obs = Observer(latitude=lat, longitude=lon)
        return solar_noon(date_obj, obs, tzname)
    except Exception:
        pass
    # Astral legacy
    try:
        from astral.location import Location
        loc = Location()
        loc.latitude = lat; loc.longitude = lon
        loc.timezone = tzinfo.zone if tzinfo else "UTC"
        return loc.solar_noon(date_obj)
    except Exception:
        pass
    # Fallback : 12:00 locale
    try:
        naive = dt.datetime.combine(date_obj, dt.time(12, 0))
        return tzinfo.localize(naive) if tzinfo else naive
    except Exception:
        return None

def compute_solar_azimuth_deg(lat, lon, when_dt):
    # pvlib -> astral -> None
    try:
        import pandas as pd
        import pvlib
        if when_dt.tzinfo is None and tzinfo:
            when_dt = tzinfo.localize(when_dt)
        times = pd.DatetimeIndex([when_dt])
        az = float(pvlib.solarposition.get_solarposition(times, lat, lon)["azimuth"].iloc[0])
        return az % 360.0
    except Exception:
        pass
    try:
        from astral import sun as astral_sun
        from astral import Observer
        when_utc = when_dt.astimezone(dt.timezone.utc) if when_dt.tzinfo else when_dt
        obs = Observer(latitude=lat, longitude=lon)
        az = float(astral_sun.azimuth(obs, when_utc))
        return az % 360.0
    except Exception:
        pass
    return None

# Datetime d'évaluation pour le SOLEIL
if use_solar_noon:
    when_local = compute_solar_noon(lat, lon, date_sel, tzinfo)
else:
    naive = dt.datetime.combine(date_sel, time_sel)
    when_local = tzinfo.localize(naive) if (tzinfo and naive.tzinfo is None) else naive

auto_solar_az = compute_solar_azimuth_deg(lat, lon, when_local) if when_local else None
if auto_solar_az is not None:
    st.info(f"Azimut **solaire** (info) : {auto_solar_az:.2f}° — {when_local.strftime('%Y-%m-%d %H:%M %Z')}")
else:
    st.caption("Azimut solaire indisponible (librairies manquantes).")

# ====== Orientation & conditions (MUR) ======
# ⚠️ On NE préremplit PAS l'azimut du mur avec l'azimut solaire.
azimuth_default = float(st.session_state.get("azimuth", 151.22))

col1, col2, col3, col4 = st.columns(4)
with col1:
    azimuth = st.number_input(
        "Azimut du mur (°)",
        value=azimuth_default,
        min_value=0.0, max_value=359.99, step=0.01,
        help="0–359.99°, depuis le Nord. 151° ≈ Sud-Sud-Est."
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
        help="Pertes d’irradiation dues aux obstacles."
    )
with col4:
    wind_ref = st.number_input(
        "Vent (m/s – indicatif)", value=float(st.session_state.get("wind_ref", 3.0)),
        min_value=0.0, step=0.5
    )

st.session_state.update({"azimuth": float(azimuth), "tilt": float(tilt), "shading": int(shading), "wind_ref": float(wind_ref)})

def azimut_cardinal(a):
    labels = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSO","SO","OSO","O","ONO","NO","NNO"]
    idx = int((a % 360) / 22.5 + 0.5) % 16
    return labels[idx]

st.caption(
    f"🧭 **Azimut MUR** : {azimuth:.2f}° ({azimut_cardinal(azimuth)}) • "
    f"📐 **Inclinaison** : {tilt:.0f}° • "
    f"🌫️ **Ombrage** : {shading}% • "
    f"💨 **Vent** : {wind_ref:.1f} m/s"
)

# Flèche d’azimut sur carte (direction façade)
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

site_layer = pdk.Layer("ScatterplotLayer", data=point_df, get_position='[lon, lat]', get_radius=6, radius_scale=10, pickable=True)
arrow_layer = pdk.Layer("PathLayer", data=[{"path": line_df[["lon","lat"]].values.tolist()}], get_width=4, width_min_pixels=2, pickable=False)

view_state = pdk.ViewState(longitude=lon, latitude=lat, zoom=15, pitch=45, bearing=float(azimuth))
tooltip = {"html": "<b>Site</b><br/>Lat: {lat}<br/>Lon: {lon}", "style": {"color": "white"}}
mapbox_key = os.getenv("MAPBOX_API_KEY", "")

if mapbox_key:
    pdk.settings.mapbox_api_key = mapbox_key
    deck = pdk.Deck(map_provider="mapbox", map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state, layers=[site_layer, arrow_layer], tooltip=tooltip)
else:
    deck = pdk.Deck(map_provider="carto", map_style="light",
                    initial_view_state=view_state, layers=[site_layer, arrow_layer], tooltip=tooltip)

st.pydeck_chart(deck, use_container_width=True)


# =========================================================
# SECTION 2 — PARAMÈTRES CLIMATIQUES (PRÉREMPLI TYPE RETSCREEN)
# =========================================================
st.header("2) Paramètres climatiques")

# ---------- 1) Dépendances & helpers ----------
try:
    from meteostat import Stations, Normals  # Normales mensuelles 1991–2020
    _HAS_METEOSTAT = True
except Exception:
    _HAS_METEOSTAT = False

MOIS_FR = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

def compute_degree_days(df, base_heat=18.0, base_cool=10.0, year=None):
    year = year or date.today().year
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    T = np.asarray(df["Temp. air (°C)"], dtype=float)
    out = df.copy()
    out["DD18 (°C·j)"] = np.round(np.maximum(0.0, base_heat - T) * days, 0)
    out["DD10 (°C·j)"] = np.round(np.maximum(0.0, T - base_cool) * days, 0)
    return out

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_climate_normals_by_coords(lat: float, lon: float):
    """Retourne un DF mensuel normalisé (1991–2020) via Meteostat pour (lat,lon).
       Colonnes: Mois, Temp. air (°C), HR (%), Précip. (mm), Vent (m/s), Pression (kPa).
       Rayonnement & T° sol non fournis (optionnels, laissés vides si non dispo)."""
    if not _HAS_METEOSTAT:
        return None, None  # pas de lib -> pas d’auto

    # Trouver la station la plus proche
    stns = Stations().nearby(lat, lon).fetch(3)
    if stns.empty:
        return None, None
    stn_id = stns.index[0]
    stn_meta = stns.iloc[0].to_dict()

    # Normales mensuelles (1991–2020)
    try:
        normals = Normals(stn_id, start=1991, end=2020).fetch()
    except Exception:
        return None, stn_meta

    # Colonnes possibles: tavg (°C), prcp (mm), pres (hPa), rhum (%), wspd (km/h) selon station
    df = pd.DataFrame({
        "Mois": MOIS_FR,
        "Temp. air (°C)": normals.get("tavg", pd.Series([np.nan]*12)).values,
        "HR (%)":        normals.get("rhum", pd.Series([np.nan]*12)).values,
        "Précip. (mm)":  normals.get("prcp", pd.Series([np.nan]*12)).values,
        "Vent (m/s)":    (normals.get("wspd", pd.Series([np.nan]*12)) / 3.6).values,  # km/h -> m/s
        "Pression (kPa)":(normals.get("pres", pd.Series([np.nan]*12)) / 10.0).values, # hPa -> kPa
        # Champs optionnels si tu veux les compléter ailleurs:
        "Rayon. horiz. (kWh/m²/j)": [np.nan]*12,
        "T° sol (°C)":              [np.nan]*12,
    })

    # Degrés-jours
    df = compute_degree_days(df)
    return df, stn_meta

# ---------- 2) UI ----------
st.header("2) Paramètres climatiques")

# En-tête méta (manuels mais on peut les auto-renseigner si on a la donnée)
colh1, colh2, colh3 = st.columns(3)
with colh1:
    zone_clim = st.selectbox(
        "Zone climatique",
        options=["1 - Très chaud","2 - Chaud","3 - Tempéré chaud","4 - Tempéré",
                 "5 - Tempéré froid","6 - Froid","7 - Très froid","8 - Arctique"],
        index=6,
        help="Peut être ajusté automatiquement plus tard à partir des DD18."
    )
with colh2:
    elevation_m = st.number_input("Élévation (m)", value=75.0, step=1.0)
with colh3:
    amp_sol = st.number_input("Amplitude des T° du sol (°C)", value=24.2, step=0.1)

colt1, colt2, colt3 = st.columns(3)
with colt1:
    t_ext_chauff = st.number_input("T° ext. chauffage (°C)", value=-23.6, step=0.1)
with colt2:
    t_ext_clim = st.number_input("T° ext. climatisation (°C)", value=27.3, step=0.1)
with colt3:
    vent_ref = st.number_input("Vitesse du vent réf. (m/s)", value=4.0, step=0.1)

# Préréglage local embarqué (saint-augustin-de-desmaures, par ex.)
DEFAULT_CLIMATE_SADM = {
    "Mois": MOIS_FR,
    "Temp. air (°C)": [-12.4, -11.0, -4.6, 3.3, 10.8, 16.3, 19.1, 17.2, 12.5, 6.5, 0.5, -9.1],
    "HR (%)": [69.1, 66.8, 66.1, 64.4, 64.0, 68.8, 73.6, 74.1, 75.9, 74.1, 74.1, 75.0],
    "Précip. (mm)": [68.29, 64.52, 79.27, 81.89, 96.29, 119.33, 122.19, 114.88, 102.99, 112.61, 101.26, 92.38],
    "Rayon. horiz. (kWh/m²/j)": [1.62, 2.66, 3.92, 4.92, 5.76, 5.30, 5.65, 4.43, 3.49, 2.61, 1.85, 1.52],
    "Pression (kPa)": [100.6, 100.6, 100.5, 100.5, 100.6, 100.5, 100.4, 100.5, 100.7, 100.8, 100.7, 100.7],
    "Vent (m/s)": [4.7, 4.7, 4.7, 4.5, 4.2, 3.6, 3.1, 3.4, 3.3, 3.9, 4.3, 4.5],
    "T° sol (°C)": [-14.6, -12.7, -6.7, 2.5, 10.0, 16.8, 19.0, 18.2, 13.0, 5.4, -1.9, -10.3],
}
DEFAULT_CLIMATE_SADM = compute_degree_days(pd.DataFrame(DEFAULT_CLIMATE_SADM))

# Source des données
source_climat = st.radio(
    "Source des données climatiques :",
    ["Auto (coordonnées → normales)", "Préréglage local (valeurs type)", "Manuel"],
    index=0,
    help="Auto : remplit avec la station la plus proche (Meteostat). Préréglage : tableau figé. Manuel : tu édites."
)

# ---------- 3) Détection de changement de site ----------
site_key = f"{round(float(st.session_state.get('lat', 0.0)), 4)},{round(float(st.session_state.get('lon', 0.0)), 4)}"
if "climate_site_key" not in st.session_state:
    st.session_state["climate_site_key"] = None

site_changed = (st.session_state["climate_site_key"] != site_key)
if site_changed and source_climat.startswith("Auto"):
    # force un refresh des data auto
    st.session_state.pop("climat_mensuel_df", None)

# ---------- 4) Construction du DataFrame ----------
if source_climat.startswith("Auto"):
    df_auto, stn_meta = fetch_climate_normals_by_coords(
        float(st.session_state.get("lat", 0.0)),
        float(st.session_state.get("lon", 0.0))
    )
    if df_auto is not None:
        base_df = df_auto
        st.session_state["climate_site_key"] = site_key
        if stn_meta:
            st.caption(f"📡 Station Meteostat la plus proche : **{stn_meta.get('name','?')}** ({stn_meta.get('country','')})")
    else:
        st.warning("Impossible d’obtenir des normales automatiques (lib manquante ou station indisponible). Préréglage utilisé.")
        base_df = DEFAULT_CLIMATE_SADM.copy()

elif source_climat.startswith("Préréglage"):
    base_df = DEFAULT_CLIMATE_SADM.copy()

else:  # Manuel
    if "climat_mensuel_df" not in st.session_state:
        st.session_state["climat_mensuel_df"] = DEFAULT_CLIMATE_SADM.copy()
    base_df = st.session_state["climat_mensuel_df"]

# ---------- 5) Éditeur ----------
# ! évite num_rows="fixed" (incompatible selon versions)
clim_df = st.data_editor(
    base_df,
    key="clim_editor",
    use_container_width=True,
    hide_index=True,
)

# Recalcul DD si l’utilisateur a modifié les T°
if source_climat != "Préréglage local (valeurs type)":
    try:
        clim_df = compute_degree_days(clim_df)
    except Exception:
        pass

# Sauvegarde état
st.session_state["climat_mensuel_df"] = clim_df

# ---------- 6) Méta & synthèse ----------
st.session_state["climat_meta"] = {
    "latitude": float(st.session_state.get("lat", 0.0)),
    "longitude": float(st.session_state.get("lon", 0.0)),
    "zone_climatique": zone_clim,
    "elevation_m": elevation_m,
    "t_ext_calc_chauffage_C": t_ext_chauff,
    "t_ext_calc_clim_C": t_ext_clim,
    "amplitude_sol_C": amp_sol,
    "vent_ref_ms": vent_ref,
}

with st.expander("Synthèse annuelle"):
    moy_air = clim_df["Temp. air (°C)"].mean(skipna=True)
    moy_vent = clim_df["Vent (m/s)"].mean(skipna=True) if "Vent (m/s)" in clim_df else np.nan
    moy_ray = clim_df["Rayon. horiz. (kWh/m²/j)"].mean(skipna=True) if "Rayon. horiz. (kWh/m²/j)" in clim_df else np.nan
    sum_dd18 = clim_df["DD18 (°C·j)"].sum(skipna=True) if "DD18 (°C·j)" in clim_df else np.nan
    sum_dd10 = clim_df["DD10 (°C·j)"].sum(skipna=True) if "DD10 (°C·j)" in clim_df else np.nan
    st.write(
        f"• **T° air moyenne**: {moy_air:.1f} °C | "
        f"**Vent moyen**: {moy_vent:.1f} m/s | "
        f"**Rayonnement moyen**: {moy_ray:.2f} kWh/m²/j | "
        f"**DD18 annuels**: {sum_dd18:.0f} °C·j | "
        f"**DD10 annuels**: {sum_dd10:.0f} °C·j"
    )
    
# ==========================
# SECTION 3 – Systéme de chauffage solaire de l'air 
# ==========================
st.header("3) Système de chauffage solaire de l’air")
unit_mode = st.radio("Unités", ["Métrique (SI)", "Impériales"], horizontal=True)

# Helpers conversion
FT2_PER_M2 = 10.7639
KBTU_PER_KWH = 3.412/1.0
CFM_PER_LPS = 2.11888

def m2_to_ft2(x): return x * FT2_PER_M2
def ft2_to_m2(x): return x / FT2_PER_M2
def kwhm2_to_kbtuft2(x): return x * 0.317097  # 1 kWh/m² ≈ 0.317 kBtu/ft²
def kbtuft2_to_kwhm2(x): return x / 0.317097
def lps_to_cfm(x): return x * CFM_PER_LPS
def cfm_to_lps(x): return x / CFM_PER_LPS

# -- Positionnement solaire (RETScreen : Fixe) --
with st.expander("Évaluation des ressources (positionnement solaire)", expanded=True):
    st.markdown("**Système de positionnement solaire** : *Fixe* (mur)")
    # On réutilise tilt & azimuth déjà saisis aux sections précédentes
    col_pos1, col_pos2 = st.columns(2)
    with col_pos1:
        st.number_input("Inclinaison (°)", value=float(tilt), key="tilt_echo", help="0°=horizontal, 90°=vertical", disabled=True)
    with col_pos2:
        st.number_input("Azimut (°)", value=float(azimuth), key="azimuth_echo", help="0°=Nord; 180°=Sud", disabled=True)

# -- Source d'irradiation mensuelle/annuelle --
mode_meteo = st.radio(
    "Source d’irradiation sur **plan du mur** (kWh/m²·an ou mensuel)",
    ["Saisie rapide (annuelle)", "Tableau mensuel (upload RETScreen .csv/.xlsx)"]
)

annual_kwh_m2 = None
monthly_df = None

if mode_meteo == "Saisie rapide (annuelle)":
    annual_kwh_m2 = st.number_input(
        "Irradiation **annuelle** sur plan du mur (kWh/m²·an)",
        value=350.0, min_value=50.0, max_value=1500.0, step=10.0,
        help="Valeur sur plan **vertical** avec ton azimut réel. Idéalement issue de RETScreen/mesures."
    )
    st.caption("Astuce : au Québec, mur S–SSE typique : ~300–500 kWh/m²·an sur plan vertical. Utilise RETScreen si possible.")
else:
    up = st.file_uploader("Importer un **mensuel RETScreen** (colonnes Mois, kWh/m² sur plan du mur)", type=["csv", "xlsx"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                monthly_df = pd.read_csv(up)
            else:
                monthly_df = pd.read_excel(up)
            # Normalisation colonnes
            monthly_df.columns = [str(c).strip().lower() for c in monthly_df.columns]
            # Détection colonnes
            mcol = next((c for c in monthly_df.columns if ("mois" in c) or ("month" in c)), None)
            vcol = next((c for c in monthly_df.columns if ("kwh" in c) and ("/m²" in c or "m2" in c or "per m2" in c or "per m²" in c)), None)
            if vcol is None:
                # fallback si le titre est "kwh/m2" sans slash m² détectable
                vcol = next((c for c in monthly_df.columns if "kwh" in c), None)

            if mcol is None or vcol is None:
                st.error("Le fichier doit contenir une colonne **Mois** et une colonne d’irradiation **kWh/m²**.")
            else:
                dfm = monthly_df[[mcol, vcol]].copy()
                dfm.columns = ["Mois", "kWh/m²"]

                # Tri des mois si 12 lignes
                mois_ordre = ["jan", "fév", "fev", "mar", "avr", "mai", "jun", "jui", "aoû", "aou", "sep", "oct", "nov", "déc", "dec"]
                if len(dfm) == 12:
                    def key_mois(x):
                        s = str(x).strip().lower()[:3]
                        # mapping pour juin/juil en fr/en
                        s = s.replace("jun", "jun").replace("jui", "jui")
                        for i, m in enumerate(mois_ordre):
                            if s == m:
                                return i
                        # mapping anglais
                        en = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
                        if s in en:
                            return en.index(s)
                        return 99
                    dfm["__k"] = dfm["Mois"].apply(key_mois)
                    dfm = dfm.sort_values("__k").drop(columns="__k").reset_index(drop=True)

                monthly_df = dfm
                annual_kwh_m2 = float(monthly_df["kWh/m²"].sum())
                st.success(f"Irradiation **annuelle** reconstituée : **{annual_kwh_m2:,.0f} kWh/m²·an**")

                # Graphique mensuel
                fig = plt.figure(figsize=(6,3))
                plt.bar(monthly_df["Mois"].astype(str), monthly_df["kWh/m²"])
                plt.ylabel("kWh/m²")
                plt.title("Irradiation mensuelle sur le plan du mur")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

# -- Portion d'utilisation par mois (RETScreen : 100% par défaut) --
with st.expander("Portion d'utilisation par mois (cas proposé) — %", expanded=False):
    # Table par défaut 12 mois à 100 %
    mois_labels = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"]
    usage_default = pd.DataFrame({"Mois": mois_labels, "Utilisation %": [100]*12})

    # ✅ Config colonne robuste (borne 0–100)
    col_config = {
        "Utilisation %": st.column_config.NumberColumn(
            "Utilisation %", min_value=0, max_value=100, step=1, format="%d"
        )
    }

    # ❌ retire num_rows / help (source fréquente de TypeError)
    usage_df = st.data_editor(
        usage_default,
        hide_index=True,
        use_container_width=True,
        column_config=col_config,
    )

    # Sécurise les entrées utilisateur
    usage_df["Utilisation %"] = pd.to_numeric(usage_df["Utilisation %"], errors="coerce").fillna(0)
    usage_df["Utilisation %"] = usage_df["Utilisation %"].clip(lower=0, upper=100)


# -- Paramètres du capteur (style RETScreen) --
with st.expander("Paramètres du capteur solaire à air", expanded=True):
    TYPES = {
        "Mur solaire sans vitrage (UTSC)": {
            "absorptivite": 0.94,
            "facteur_correctif": 1.00,
            "comment": "Mur solaire perforé, tirage mécanique. ΔT élevé par temps ensoleillé."
        },
        "Capteur à air vitré": {
            "absorptivite": 0.95,
            "facteur_correctif": 1.05,
            "comment": "Caisson vitré + absorbeur. Meilleur en intersaison, pertes nocturnes ↑."
        },
        "Vitré + absorbeur sélectif": {
            "absorptivite": 0.96,
            "facteur_correctif": 1.10,
            "comment": "Absorbeur sélectif, meilleur à faible éclairement, coût ↑."
        },
    }

    type_capteur = st.selectbox("Type de capteur", list(TYPES.keys()), index=0)
    defaults = TYPES[type_capteur]

    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        absorptivite = st.number_input(
            "Absorptivité du capteur", min_value=0.80, max_value=0.99,
            value=float(defaults["absorptivite"]), step=0.01
        )
        couleur = st.selectbox("Couleur/finition", ["Noir", "Anthracite", "Autre"], index=0)
    with colc2:
        facteur_correctif = st.number_input(
            "Facteur correctif global (adim.)",
            min_value=0.50, max_value=2.00, value=float(defaults["facteur_correctif"]), step=0.01,
            help="Facteur multiplicatif pour caler le modèle (ombrage résiduel, pertes/inconnues, gains d’aspiration)."
        )
        if facteur_correctif > 1.20:
            st.warning("Facteur > 1.20 : vérifie et documente la raison (aspiration, mesures, etc.).")
    with colc3:
        surface_m2 = st.number_input(
            "Surface de capteur (m²)", min_value=1.0, value=150.0, step=1.0,
            help="Surface nette exposée."
        )

    # Garantir l’existence des paramètres d’ombrage/vent
    st.session_state.setdefault("ombrage_saison", 10)
    st.session_state.setdefault("atten_vent", 0)

    ombrage_saison = st.slider(
        "Ombrage sur le capteur – période d'utilisation (%)",
        0, 90, int(st.session_state["ombrage_saison"]), step=1
    )
    st.session_state["ombrage_saison"] = ombrage_saison

    atten_vent = st.slider(
        "Atténuation des vents – saison d'utilisation (%)",
        0, 50, int(st.session_state["atten_vent"]), step=1,
        help="Pertes supplémentaires dues au vent."
    )
    st.session_state["atten_vent"] = atten_vent

    st.caption(f"ℹ️ {defaults['comment']}")

# -- Application des portions d'utilisation mensuelles sur l'irradiation (si mensuelle fournie) --
# kWh/m² utile = irradiation * (utilisation%/100) * (1 - ombrage) * (1 - atténuation vent)

# Sécurise usage_df (si non défini plus haut)
if "usage_df" not in locals():
    mois_labels = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"]
    usage_df = pd.DataFrame({"Mois": mois_labels, "Utilisation %": [100]*12})

# Pertes (bornées 0–1)
perte_ombrage = max(0.0, 1.0 - ombrage_saison/100.0)
perte_vent    = max(0.0, 1.0 - atten_vent/100.0)
facteur_pertes = perte_ombrage * perte_vent

monthly_used = None
if ("monthly_df" in locals()) and (monthly_df is not None) and ("kWh/m²" in monthly_df.columns):
    # Normalisation robuste des mois FR/EN
    def _normalize_mois(x: str) -> str:
        s = str(x).strip().lower()[:3]
        mapping = {
            # Français
            "jan":"Jan", "fév":"Fév", "fev":"Fév", "mar":"Mar", "avr":"Avr", "mai":"Mai",
            "jui":"Juil", "jun":"Juin", "aoû":"Août", "aou":"Août", "sep":"Sep", "oct":"Oct", "nov":"Nov", "déc":"Déc", "dec":"Déc",
            # Anglais
            "feb":"Fév", "apr":"Avr", "may":"Mai", "jul":"Juil", "aug":"Août"
        }
        return mapping.get(s, s.title())

    mdf = monthly_df.copy()
    mdf["Mois"] = mdf["Mois"].apply(_normalize_mois)

    # Merge avec usage (%)
    tmp = pd.merge(mdf, usage_df, on="Mois", how="left")
    tmp["Utilisation %"] = pd.to_numeric(tmp["Utilisation %"], errors="coerce").fillna(100).clip(0, 100)

    # kWh/m² utile
    tmp["kWh/m² utile"] = tmp["kWh/m²"] * (tmp["Utilisation %"]/100.0) * facteur_pertes
    monthly_used = tmp[["Mois","kWh/m²","Utilisation %","kWh/m² utile"]]

    # Graphique utile
    fig2 = plt.figure(figsize=(6,3))
    plt.bar(monthly_used["Mois"], monthly_used["kWh/m² utile"])
    plt.ylabel("kWh/m² utile")
    plt.title("Irradiation utile (pondérée utilisation & pertes)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

# -- Irradiation annuelle "utile" sur m²
if monthly_used is not None:
    annual_kwh_m2_utile = float(monthly_used["kWh/m² utile"].sum())
elif annual_kwh_m2 is not None:
    annual_kwh_m2_utile = float(annual_kwh_m2) * facteur_pertes  # si pas de saison mensuelle, on applique pertes globales
else:
    annual_kwh_m2_utile = None

# --- Sortie synthèse bloc 3 ---
st.markdown("### Synthèse Bloc 3")
colS1, colS2, colS3 = st.columns(3)
with colS1:
    st.metric("Irradiation annuelle (sur plan)", f"{(annual_kwh_m2 or 0):,.0f} kWh/m²·an")
with colS2:
    st.metric("Irradiation annuelle **utile**", f"{(annual_kwh_m2_utile or 0):,.0f} kWh/m²·an")
with colS3:
    st.metric("Surface capteur", f"{surface_m2:,.0f} m²")

# Tu auras ensuite : énergie solaire reçue utile ~ annual_kwh_m2_utile * surface_m2 (avant rendement aéraulique/thermique).
energie_solaire_utile_kwh = (annual_kwh_m2_utile or 0) * surface_m2
st.caption(f"Énergie solaire utile reçue (avant conversion aéraulique/ΔT) ≈ **{energie_solaire_utile_kwh:,.0f} kWh/an**")

# ==========================
# SECTION 4 – PERFORMANCE COLLECTEUR (UTC / mur solaire)
# ==========================
st.header("4) Performance – Mur solaire (transpiré non vitré)")
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
























