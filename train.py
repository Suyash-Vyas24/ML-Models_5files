import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def generate_realistic_ev_data(n_samples=5000):
    np.random.seed(42)
    
    # Generate fake sensor data
    soc = np.random.uniform(10, 100, n_samples)
    speed = np.random.uniform(0, 120, n_samples)
    ambient_temp = np.random.uniform(5, 45, n_samples)
    current = (speed / 15) ** 2 + np.random.normal(5, 2, n_samples)
    current = np.clip(current, 0, 150) 
    batt_temp = ambient_temp + (current * 0.15) + np.random.normal(0, 1, n_samples)
    
    cycles = np.random.uniform(10, 1500, n_samples)
    age_months = np.random.uniform(1, 60, n_samples)
    fast_charge_ratio = np.random.uniform(0.0, 1.0, n_samples)
    
    # Calculate Range
    base_range = soc * 3.0 
    speed_penalty = (speed / 100) * 45 
    temp_penalty = np.where(batt_temp < 15, (15 - batt_temp) * 1.2, 
                   np.where(batt_temp > 35, (batt_temp - 35) * 1.5, 0))
    range_km = base_range - speed_penalty - temp_penalty + np.random.normal(0, 2, n_samples)
    range_km = np.clip(range_km, 0, 300)
    
    # Calculate Health
    soh = 100 - (cycles * 0.012) - (age_months * 0.08) - (fast_charge_ratio * cycles * 0.005)
    soh = soh + np.random.normal(0, 1, n_samples)
    soh = np.clip(soh, 40, 100)
    
    return pd.DataFrame({
        'soc_percent': soc, 'speed_kmh': speed, 'current_a': current, 
        'batt_temp_c': batt_temp, 'cycles': cycles, 'age_months': age_months,
        'fast_charge_ratio': fast_charge_ratio, 'range_km': range_km, 'soh_percent': soh
    })

print("1. Generating EV data...")
df = generate_realistic_ev_data(5000)

print("2. Training AI Models (This takes a few seconds)...")
# Train Range Model
X_range = df[['soc_percent', 'speed_kmh', 'current_a', 'batt_temp_c']]
y_range = df['range_km']
range_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
range_model.fit(X_range, y_range)

# Train Health Model
X_soh = df[['cycles', 'age_months', 'fast_charge_ratio', 'batt_temp_c']]
y_soh = df['soh_percent']
soh_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
soh_model.fit(X_soh, y_soh)

print("3. Saving Models...")
joblib.dump(range_model, 'ev_range_model.pkl')
joblib.dump(soh_model, 'ev_soh_model.pkl')
print("✅ SUCCESS! The .pkl files have been created in your folder!")