import geopandas as gpd
import joblib
from flask import Flask, render_template, jsonify, request
import folium
from folium import Choropleth
from sklearn.preprocessing import StandardScaler
from pyngrok import ngrok
import numpy as np

app = Flask(__name__)

# Ensure Flask runs on all network interfaces for ngrok
public_url = ngrok.connect(5000)
print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:5000")

# Paths to model, scaler, and shapefile
model_path = "/content/drive/MyDrive/Colab Notebooks/Cholera Modeling and Prediction/Yobe state/models/random-forest-model.joblib"
scaler_path = "/content/drive/MyDrive/Colab Notebooks/Cholera Modeling and Prediction/Yobe state/models/scaler_rf.joblib"
data_path = "/content/drive/MyDrive/Colab Notebooks/Cholera Modeling and Prediction/Yobe state/Population_Cholera.shp"

# Load trained model and scaler
trained_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load shapefile (GeoDataFrame)
prediction_data = gpd.read_file(data_path)

# Ensure CRS is set to EPSG:4326 (WGS84)
if prediction_data.crs.to_epsg() != 4326:
    prediction_data = prediction_data.to_crs(epsg=4326)

# Features for prediction
all_features = ['Aspect', 'Elevatn', 'builtupr', 'LST', 'LULCC', 'NDVI', 'NDWI', 'PopDnsty', 'Poverty', 'Prcpittn', 'Slope', 'rwi']
display_features = ['Aspect', 'Elevation', 'Built-up Area', 'LST', 'Land use/Cover', 'NDVI', 'NDWI', 'Pop Density', 'Poverty', 'Precipitation', 'Slope', 'Relative Wealth Index']

# Default: all features selected
selected_features = all_features.copy()

# Base year
base_year = 2024

# Function to modify features based on future years
def adjust_for_future(X_pred, year):
    """Adjusts features like population and climate conditions for the selected year."""
    year_diff = year - base_year  # Calculate how far into the future

    if year_diff > 0:
        X_pred['PopDnsty'] *= (1 + 0.02 * year_diff)  # 2% yearly population growth
        X_pred['Prcpittn'] *= (1 + 0.01 * year_diff)  # 1% increase in precipitation
        X_pred['LST'] += 0.5 * year_diff  # Temperature rise

    return X_pred

# Function to update the map
def update_map(selected_features, year):
    """Generates and updates the map based on selected features and future year changes."""
    X_pred = prediction_data[all_features].copy()

    # Set unselected features to mean values instead of zero
    for feature in all_features:
        if feature not in selected_features:
            X_pred[feature] = X_pred[feature].mean()  

    # Adjust for future years
    X_pred = adjust_for_future(X_pred, year)

    # Handle missing values
    X_pred = X_pred.fillna(X_pred.mean())

    # Scale features
    X_pred_scaled = scaler.transform(X_pred)

    # Make predictions
    predictions = trained_model.predict(X_pred_scaled)
    prediction_data['pred_cases'] = predictions.astype(int)

    # Ensure 'ward_name' exists in the data
    if 'ward_name' not in prediction_data.columns:
        prediction_data['ward_name'] = [f"Ward {i}" for i in range(len(prediction_data))]

    # Map center
    center = prediction_data.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=8)

    # Add choropleth layer
    Choropleth(
        geo_data=prediction_data,
        data=prediction_data,
        columns=['ward_name', 'pred_cases'],
        key_on='feature.properties.ward_name',
        fill_color='YlOrRd',
        fill_opacity=1.0,
        line_opacity=0.2,
        legend_name=f'Predicted Cholera Cases in {year}'
    ).add_to(m)

    # Add tooltips for interactive display
    for _, row in prediction_data.iterrows():
        folium.GeoJson(
            row.geometry,
            tooltip=folium.Tooltip(
                f"<b>Predicted Cases:</b> {row['pred_cases']}<br>"
                f"<b>Ward:</b> {row['ward_name']}<br>"
                f"<b>LGA:</b> {row.get('lga_name', 'Unknown')}<br>"
                f"<b>Year:</b> {year}"
            ),
        ).add_to(m)

    return m._repr_html_()

@app.route('/')
def index():
    """Renders the main page with the initial map."""
    year = base_year  # Default starting year (2024)
    map_html = update_map(selected_features, year)
    return render_template('index.html', map_html=map_html, all_features=display_features, year=year)

@app.route('/update', methods=['POST'])
def update():
    """Handles feature and year updates dynamically."""
    data = request.json
    selected_features = data.get('selected_features', all_features)  # Ensure default features
    year = int(data.get('year', base_year))  # Get selected year
    map_html = update_map(selected_features, year)
    return jsonify(map_html=map_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
