import streamlit as st
import pydeck as pdk
import pandas as pd

# Import from our modularized files
from config import ZONE_COORDS, COUNTRY_NAMES
from map_builder import generate_map_nodes, generate_flow_arcs
from charts import plot_price_vs_weather, plot_generation_mix
from data_loader import (
    load_all_grid_flows, load_generation_mix,
    load_all_weather, load_total_load, load_spot_price
)

@st.cache_data
def fetch_all_data() -> tuple:
    return (
        load_all_grid_flows(), load_generation_mix(),
        load_all_weather(), load_total_load(), load_spot_price()
    )

def get_closest_flows(flows_data: pd.DataFrame, target_timestamp: pd.Timestamp) -> pd.DataFrame:
    hourly_flows = flows_data[flows_data['time'] == target_timestamp].copy()
    if hourly_flows.empty:
        nearest_index = (flows_data['time'] - target_timestamp).abs().argmin()
        hourly_flows = flows_data.iloc[nearest_index:nearest_index+1].copy()
    return hourly_flows

def render_country_analytics(country: str, current_timestamp: pd.Timestamp, weather_data: pd.DataFrame, price_data: pd.DataFrame, gen_data: pd.DataFrame):
    start_time = current_timestamp - pd.Timedelta(hours=12)
    end_time = current_timestamp + pd.Timedelta(hours=12)

    country_weather = weather_data[(weather_data['zone'] == country) & (weather_data['time'].between(start_time, end_time))].copy()
    country_price = price_data[(price_data['zone'] == country) & (price_data['time'].between(start_time, end_time))].copy()
    country_generation = gen_data[(gen_data['zone'] == country) & (gen_data['time'].between(start_time, end_time))].copy()

    if country_weather.empty or country_price.empty:
        st.warning(f"Insufficient data to run analysis for {COUNTRY_NAMES.get(country, country)}.")
        return

    # 1. TOP RIGHT: Generation Mix
    if not country_generation.empty:
        plot_generation_mix(country_generation, current_timestamp)
    else:
        st.info("No generation data available for this timeframe.")

    # 2. BOTTOM RIGHT: Weather Selector & Small Multiples
    merged_metrics = country_price[['time', 'price_eur_mwh']].copy()
    weather_cols = [
        'wind_speed_100m (km/h)', 
        'cloud_cover (%)', 
        'temperature_2m (°C)', 
        'relative_humidity_2m (%)', 
        'precipitation (mm)'
    ]

    for col in weather_cols:
        if col in country_weather.columns:
            merged_metrics = merged_metrics.merge(country_weather[['time', col]], on='time', how='left')

    merged_metrics = merged_metrics.sort_values('time').dropna(subset=['price_eur_mwh'])
    
    selected_weather = st.multiselect(
        "Compare Weather Conditions to Price (Max 2):", 
        options=weather_cols,
        default=[weather_cols[0], weather_cols[1]],
        max_selections=2,
        format_func=lambda x: x.split('(')[0].replace('_', ' ').strip().title(),
        key=f"compare_{country}"
    )

    if selected_weather:
        plot_price_vs_weather(merged_metrics, selected_weather, current_timestamp)
    else:
        st.info("Select at least one weather condition to view the correlation.")


def main():
    st.set_page_config(layout="wide", page_title="EU Grid Matrix", page_icon="⚡")
    
    st.markdown("""
        <style>
            body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
                overflow: hidden !important;
                width: 100vw !important;
                height: 100vh !important;
            }
            .block-container { 
                padding-top: 1rem; padding-bottom: 0rem; 
                padding-left: 2rem; padding-right: 2rem; 
                max-height: 100vh;
                max-width: 100vw;
                overflow-x: hidden !important;
            }
            header { display: none !important; }
            footer { display: none !important; }
        </style>
        """, unsafe_allow_html=True)

    flows_data, gen_data, weather_data, load_data, price_data = fetch_all_data()
    all_times = sorted(pd.to_datetime(flows_data['time']).unique())
    time_min, time_max = all_times[0].to_pydatetime(), all_times[-1].to_pydatetime()

    st.markdown("<h3 style='margin-top: -15px; margin-bottom: 10px; text-align: center;'>Energy Across EU Dashboard</h3>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        flow_direction = st.radio("Flow Direction:", ["Show Imports/Exports", "Show Exports", "Show Imports"], horizontal=True)

        all_zones = sorted(list(ZONE_COORDS.keys()))
        selected_countries = st.multiselect(
            "Map Country Selection (Max. 6):",
            options=all_zones,
            default=["DE", "FR", "DK1"],
            max_selections=6,
            format_func=lambda x: COUNTRY_NAMES.get(x, x)
        )
        
        selected_time = st.slider("Time filtering", min_value=time_min, max_value=time_max, value=time_min, step=pd.Timedelta(hours=1), format="MM-DD HH:mm")
        current_timestamp = pd.Timestamp(selected_time)
        
        hourly_flows = get_closest_flows(flows_data, current_timestamp)
        current_timestamp = pd.Timestamp(hourly_flows['time'].values[0])
        
        map_nodes = pd.DataFrame(generate_map_nodes(selected_countries))
        if not map_nodes.empty:
            map_nodes['full_name'] = map_nodes['zone'].apply(lambda x: COUNTRY_NAMES.get(x, x))
            
        map_arcs_raw = generate_flow_arcs(hourly_flows, selected_countries, flow_direction)
        if map_arcs_raw:
            map_arcs = pd.DataFrame(map_arcs_raw)
            def format_arc_tooltip(row):
                if 'label' in row:
                    lbl = str(row['label'])
                    for code, name in COUNTRY_NAMES.items():
                        lbl = lbl.replace(code, name)
                    return lbl
                return "Energy Flow"
            map_arcs['full_name'] = map_arcs.apply(format_arc_tooltip, axis=1)
        else:
            map_arcs = None

        # --- PAUL TOL COLORS (Muted Palette) --- i love this guy since day 1
        TOL_TEAL = [68, 170, 153, 200]   # #44AA99
        TOL_ROSE = [204, 102, 119, 200]  # #CC6677
        TOL_SAND = [221, 204, 119, 200]  # #DDCC77 

        map_layers = [
            pdk.Layer(
                "ScatterplotLayer", 
                map_nodes, 
                get_position="coords", 
                get_fill_color="color", 
                get_radius="radius", 
                pickable=True, 
                auto_highlight=True, 
                highlight_color=TOL_SAND, 
                pickable_radius=12000
            )
        ]
        if map_arcs is not None and not map_arcs.empty:
            map_layers.append(pdk.Layer(
                "ArcLayer", 
                map_arcs, 
                get_source_position="source", 
                get_target_position="target", 
                get_source_color=TOL_TEAL, 
                get_target_color=TOL_ROSE, 
                get_width="value / 200", 
                pickable=True, 
                auto_highlight=True
            ))

        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", 
            initial_view_state=pdk.ViewState(latitude=50, longitude=10, zoom=3.5, pitch=45), 
            layers=map_layers, 
            height=380, 
            tooltip={"text": "{full_name}"} 
        ))

        # legends never die ;-;
        st.markdown(
            "<div style='text-align: center; font-size: 15px; margin-top: 5px;'>"
            "<span style='color:#44AA99; font-weight:bold;'>● Origin</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
            "<span style='color:#CC6677; font-weight:bold;'>● Destination</span>"
            "</div>", 
            unsafe_allow_html=True
        )

    with col_right:
        if not selected_countries:
            st.info("Start by selecting a country on the left to see energy generation and weather vs. spot prices.")
        else:
            analytics_tabs = st.tabs([f"{COUNTRY_NAMES.get(country, country)}" for country in selected_countries])
            for tab, country in zip(analytics_tabs, selected_countries):
                with tab:
                    with st.container(height=880, border=False):
                        render_country_analytics(country, current_timestamp, weather_data, price_data, gen_data)

if __name__ == "__main__":
    main()