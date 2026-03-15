import pandas as pd
import streamlit as st
import altair as alt
from config import GENERATION_COLORS

def get_broad_category(raw_col: str) -> str:
    """Smart substring matching to catch complex ENTSO-E data labels."""
    c = str(raw_col).lower()
    if 'offshore' in c: return 'Wind Offshore'
    if 'wind' in c: return 'Wind Onshore'
    if 'solar' in c: return 'Solar'
    if 'nuclear' in c: return 'Nuclear'
    if 'hydro' in c or 'pumped' in c or 'water' in c or 'river' in c: return 'Hydro'
    if 'coal' in c or 'gas' in c or 'oil' in c or 'fossil' in c or 'lignite' in c: return 'Fossil Fuels'
    if 'biomass' in c or 'waste' in c or 'wood' in c: return 'Bioenergy'
    return 'Other'

def plot_price_vs_weather(merged_data: pd.DataFrame, comparison_variables: list, current_timestamp: pd.Timestamp):
    price_color = '#E69F00' 
    weather_color = '#708090' 
    
    # 1. Selection logic: 'nearest=True' is the magic word here
    hover = alt.selection_point(
        fields=['time'],
        nearest=True,      # Finds the closest point to the cursor
        on='mouseover',
        empty=False,
        clear='mouseout'
    )

    base_x = alt.X('time:T', scale=alt.Scale(padding=5), axis=alt.Axis(title=None, tickCount=6))

    # --- THE "MAGNET" LAYER ---
    # This invisible chart captures the mouse across the whole height of the chart
    # and forces the tooltip to show for the nearest 'time'
    magnetic_selector = alt.Chart(merged_data).mark_point().encode(
        x=base_x,
        opacity=alt.value(0), # Keep it invisible
        tooltip=[
            alt.Tooltip('time:T', format='%Y-%m-%d', title='Date'),
            alt.Tooltip('time:T', format='%H:%M', title='Time'),
            alt.Tooltip('price_eur_mwh:Q', format='.2f', title='Price (€/MWh)')
        ]
    ).add_params(hover)

    # 2. PRICE CHART
    price_base = alt.Chart(merged_data).encode(x=base_x.axis(alt.Axis(labels=False, title=None))).properties(
        height=120, width=600,
        title=alt.TitleParams("Spot Price (€/MWh)", anchor='start', color=price_color, fontSize=12, dy=10)
    )
    
    price_line = price_base.mark_line(color=price_color, size=2, interpolate='monotone').encode(y=alt.Y('price_eur_mwh:Q', title=None))
    
    # Glowing point that snaps to the nearest value
    price_points = price_line.mark_circle(size=80).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    
    price_chart = alt.layer(
        price_base.mark_area(opacity=0.3, color=price_color, interpolate='monotone').encode(y=alt.Y('price_eur_mwh:Q',title=None)),
        price_line,
        price_points,
        # Vertical Trace Line
        alt.Chart(merged_data).mark_rule(color='#ffffff', opacity=0.4).encode(x='time:T').transform_filter(hover),
        # Current Time Slider Marker
        alt.Chart(pd.DataFrame({'time': [current_timestamp]})).mark_rule(color='#D55E00', size=2, strokeDash=[5]).encode(x='time:T'),
        magnetic_selector # Put this on top to catch all mouse events
    )

    # 3. WEATHER CHARTS
    weather_charts = []
    for i, var in enumerate(comparison_variables):
        if var in merged_data.columns:
            is_last = (i == len(comparison_variables) - 1)
            x_axis_config = alt.Axis(format='%H:%M', title='Time') if is_last else alt.Axis(labels=False, title=None)
            
            clean_name = var.split('(')[0].replace('_', ' ').strip().title()
            unit = var.split('(')[-1].replace(')', '') if '(' in var else ''
            
            w_base = alt.Chart(merged_data).encode(x=alt.X('time:T', scale=alt.Scale(padding=5), axis=x_axis_config)).properties(
                width=600, height=90,
                title=alt.TitleParams(f"{clean_name} ({unit})", anchor='start', color=weather_color, fontSize=11, dy=10)
            )
            
            # Weather-specific magnetic selector
            w_magnetic = alt.Chart(merged_data).mark_point().encode(
                x=base_x,
                opacity=alt.value(0),
                tooltip=[
                    alt.Tooltip('time:T', format='%Y-%m-%d', title='Date'),
                    alt.Tooltip('time:T', format='%H:%M', title='Time'),
                    alt.Tooltip(f'{var}:Q', format='.2f', title=f"{clean_name} ({unit})")
                ]
            ).add_params(hover)

            w_line = w_base.mark_line(color=weather_color, size=2, interpolate='monotone').encode(y=alt.Y(f'{var}:Q', title=None, scale=alt.Scale(zero=False)))
            w_points = w_line.mark_circle(size=80, color=weather_color).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
            
            weather_charts.append(alt.layer(
                w_line, 
                w_points,
                alt.Chart(merged_data).mark_rule(color='#ffffff', opacity=0.4).encode(x='time:T').transform_filter(hover),
                w_magnetic
            ))

    if weather_charts:
        full_stack = alt.vconcat(price_chart, *weather_charts, spacing=15).resolve_scale(x='shared')
        st.altair_chart(full_stack, use_container_width=True)
    else:
        st.altair_chart(price_chart, use_container_width=True)

def plot_generation_mix(gen_data: pd.DataFrame, current_time: pd.Timestamp, threshold: float = 0.02):
    """Plots generation mix in a wide format for the map-bottom layout."""
    st.markdown("<h4 style='text-align: center; margin-bottom: 0px;'>Energy Generation Overview</h4>", unsafe_allow_html=True)
    
    df = gen_data.pivot_table(index='time', columns='type', values='value (MW)', aggfunc='sum').fillna(0)
    df.columns = [get_broad_category(col) for col in df.columns]
    df = df.groupby(df.columns, axis=1).sum()
    df_pct = df.div(df.sum(axis=1).replace(0, 1), axis=0)
    
    max_vals = df_pct.max()
    small_sources = max_vals[max_vals < threshold].index.tolist()
    if 'Other' in small_sources: small_sources.remove('Other')
    if small_sources:
        df_pct['Other'] = df_pct.get('Other', 0.0) + df_pct[small_sources].sum(axis=1)
        df_pct = df_pct.drop(columns=small_sources)

    df_melted = df_pct.reset_index().melt(id_vars='time', var_name='Type', value_name='Percentage')
    colors = [GENERATION_COLORS.get(t, '#888888') for t in df_pct.columns]
    
    area_chart = alt.Chart(df_melted).mark_area(opacity=0.85).encode(
        x=alt.X('time:T', title='Time', axis=alt.Axis(format='%H:%M', tickCount=10)),
        y=alt.Y('Percentage:Q', stack='zero', title='Energy Sources', axis=alt.Axis(format='.0%')),
        color=alt.Color('Type:N', scale=alt.Scale(domain=list(df_pct.columns), range=colors), 
                        legend=alt.Legend(title=None, orient='top', columns=4)),
        tooltip=[
            alt.Tooltip('time:T', format='%Y-%m-%d', title='Date'),
            alt.Tooltip('time:T', format='%H:%M', title='Time'),
            'Type:N', 
            alt.Tooltip('Percentage:Q', format='.1%')
        ]
    ).properties(height=240)

    marker = alt.Chart(pd.DataFrame({'time': [current_time]})).mark_rule(color='#D55E00', size=2, strokeDash=[5]).encode(x='time:T')
    st.altair_chart(alt.layer(area_chart, marker).interactive(), use_container_width=True)