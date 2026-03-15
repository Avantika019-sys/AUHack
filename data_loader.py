import pandas as pd
import os
import glob

# Set the base directory to your data folder
DATA_DIR = "data" 

def load_zone_data(zone="DE"):
    """Loads and merges spot price, generation, and load for a given zone."""
    
    # 1. Load spot price from the 'spot-price' subfolder
    price_file = os.path.join(DATA_DIR, "spot-price", f"{zone}-spot-price.csv")
    if os.path.exists(price_file):
        df_price = pd.read_csv(price_file)
        df_price['time'] = pd.to_datetime(df_price['time'])
        df_price = df_price.rename(columns={'value (EUR/MWh)': 'spot_price'})
        df_price.set_index('time', inplace=True)
    else:
        df_price = pd.DataFrame()

    # 2. Load total load (Demand) from the 'total-load' subfolder
    load_file = os.path.join(DATA_DIR, "total-load", f"{zone}-total-load.csv")
    if os.path.exists(load_file):
        df_load = pd.read_csv(load_file)
        df_load['time'] = pd.to_datetime(df_load['time'])
        df_load = df_load.rename(columns={'value (MW)': 'total_load'})
        df_load.set_index('time', inplace=True)
    else:
        df_load = pd.DataFrame()

    # 3. Load generation (Supply) from the 'generation' subfolder
    gen_file = os.path.join(DATA_DIR, "generation", f"{zone}-generation.csv")
    if os.path.exists(gen_file):
        df_gen_raw = pd.read_csv(gen_file)
        df_gen_raw['time'] = pd.to_datetime(df_gen_raw['time'])
        # Pivot so each generation type becomes its own column
        df_gen = df_gen_raw.pivot_table(index='time', columns='type', values='value (MW)', aggfunc='sum')
    else:
        df_gen = pd.DataFrame()

    # Merge them all based on the datetime index
    df_merged = df_price.join(df_load, how='outer').join(df_gen, how='outer')
    
    # Resample to hourly (1H) to smooth out 15-min mismatches for the prototype
    df_merged = df_merged.resample('1H').mean().ffill()
    
    return df_merged

def load_flows_to_zone(target_zone="DE"):
    """Loads physical flows INTO the target zone from the 'flows' subfolder."""
    flow_file = os.path.join(DATA_DIR, "flows", f"{target_zone}-physical-flows-in.csv")
    if os.path.exists(flow_file):
        df_flows = pd.read_csv(flow_file)
        df_flows['time'] = pd.to_datetime(df_flows['time'])
        # Extract the source country from the "FR->DE" format
        df_flows['source_zone'] = df_flows['zone'].apply(lambda x: x.split('->')[0])
        
        # Pivot so each neighbor is a column
        df_flows_pivot = df_flows.pivot_table(index='time', columns='source_zone', values='value (MW)', aggfunc='sum')
        df_flows_pivot = df_flows_pivot.resample('1H').mean().ffill()
        return df_flows_pivot
        
    return pd.DataFrame()

def load_weather_data(zone="DE"):
    """Loads weather data for a zone using glob to bypass exact coordinate matching."""
    # Search for any file in the weather folder that starts with the zone and 'open-meteo'
    search_pattern = os.path.join(DATA_DIR, "weather", f"{zone}-open-meteo-*.csv")
    matching_files = glob.glob(search_pattern)

    if matching_files:
        # Weather files have 3 lines of metadata above the header, so we skip them
        df_weather = pd.read_csv(matching_files[0], skiprows=3)
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df_weather.set_index('time', inplace=True)
        # Resample to hourly just in case
        df_weather = df_weather.resample('1H').mean().ffill()
        return df_weather

    return pd.DataFrame()

def load_all_grid_flows():
    """Load all physical flows between zones and return in long format."""
    all_flows = []
    flows_dir = os.path.join(DATA_DIR, "flows")

    for flow_file in glob.glob(os.path.join(flows_dir, "*-physical-flows-in.csv")):
        df = pd.read_csv(flow_file)
        df['time'] = pd.to_datetime(df['time'])
        all_flows.append(df)

    if all_flows:
        df_all = pd.concat(all_flows, ignore_index=True)
        df_all = df_all.rename(columns={'zone': 'route', 'value (MW)': 'value'})
        df_all[['src', 'tgt']] = df_all['route'].str.split('->', expand=True)
        return df_all.sort_values('time')
    return pd.DataFrame()

def load_generation_mix():
    """Load generation data for all zones."""
    all_gen = []
    gen_dir = os.path.join(DATA_DIR, "generation")

    for gen_file in glob.glob(os.path.join(gen_dir, "*-generation.csv")):
        df = pd.read_csv(gen_file)
        df['time'] = pd.to_datetime(df['time'])
        zone = os.path.basename(gen_file).split('-')[0]
        df['zone'] = zone
        all_gen.append(df)

    if all_gen:
        return pd.concat(all_gen, ignore_index=True)
    return pd.DataFrame()

def load_total_load():
    """Load total demand for all zones."""
    all_load = []
    load_dir = os.path.join(DATA_DIR, "total-load")

    for load_file in glob.glob(os.path.join(load_dir, "*-total-load.csv")):
        df = pd.read_csv(load_file)
        df['time'] = pd.to_datetime(df['time'])
        zone = os.path.basename(load_file).split('-')[0]
        df['zone'] = zone
        df = df.rename(columns={'value (MW)': 'load_mw'})
        all_load.append(df[['time', 'zone', 'load_mw']])

    if all_load:
        return pd.concat(all_load, ignore_index=True)
    return pd.DataFrame()

def load_spot_price():
    """Load spot prices for all zones."""
    all_price = []
    price_dir = os.path.join(DATA_DIR, "spot-price")

    for price_file in glob.glob(os.path.join(price_dir, "*-spot-price.csv")):
        df = pd.read_csv(price_file)
        df['time'] = pd.to_datetime(df['time'])
        zone = os.path.basename(price_file).split('-')[0]
        df['zone'] = zone
        df = df.rename(columns={'value (EUR/MWh)': 'price_eur_mwh'})
        all_price.append(df[['time', 'zone', 'price_eur_mwh']])

    if all_price:
        return pd.concat(all_price, ignore_index=True)
    return pd.DataFrame()

def load_all_weather():
    """Load weather data for all zones."""
    all_weather = []
    weather_dir = os.path.join(DATA_DIR, "weather")

    # Map file prefixes to zone codes
    zone_mapping = {
        'AT': 'AT', 'BE': 'BE', 'CH': 'CH', 'CZ': 'CZ', 'DE': 'DE',
        'DK': 'DK1', 'FR': 'FR', 'NL': 'NL', 'PL': 'PL'
    }

    for weather_file in glob.glob(os.path.join(weather_dir, "*-open-meteo-*.csv")):
        filename = os.path.basename(weather_file)
        prefix = filename.split('-')[0]
        zone = zone_mapping.get(prefix, prefix)

        df = pd.read_csv(weather_file, skiprows=3)
        df['time'] = pd.to_datetime(df['time'])
        df['zone'] = zone
        all_weather.append(df)

    if all_weather:
        return pd.concat(all_weather, ignore_index=True)
    return pd.DataFrame()