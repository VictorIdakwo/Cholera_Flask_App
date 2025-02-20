import geopandas as gpd
import joblib
from flask import Flask, render_template, jsonify, request
import folium
from folium import Choropleth
from sklearn.preprocessing import StandardScaler
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok

app = Flask(__name__)

# Ensure Flask is running on all network interfaces (0.0.0.0) for ngrok to tunnel properly
public_url = ngrok.connect(5000)
print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:5000")

# Paths to model, scaler, and shapefile
model_path = "/content/drive/MyDrive/Colab Notebooks/Cholera Modeling and Prediction/Yobe state/models/random-forest-model.joblib"
scaler_path = "/content/drive/MyDrive/Colab Notebooks/Cholera Modeling and Prediction/Yobe state/models/scaler_rf.joblib"
data_path = "/content/drive/MyDrive/Colab Notebooks/Cholera Modeling and Prediction/Yobe state/Population_Cholera.shp"

trained_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
prediction_data = gpd.read_file(data_path)

# Ensure CRS is set to EPSG:4326 (WGS84)
if prediction_data.crs and prediction_data.crs.to_epsg() != 4326:
    prediction_data = prediction_data.to_crs(epsg=4326)

# Features for prediction
all_features = ['Aspect', 'Elevatn', 'builtupr', 'LST', 'LULCC', 'NDVI', 'NDWI', 'PopDnsty', 'Poverty', 'Prcpittn', 'Slope', 'rwi']
display_features = ['Aspect', 'Elevation', 'Built-up Area', 'LST', 'Land use/Cover', 'NDVI', 'NDWI', 'Pop Density', 'Poverty', 'Precipitation', 'Slope', 'Relative Wealth Index']

selected_features = all_features.copy()

def update_map(selected_features, year):
    X_pred = prediction_data[all_features].copy()

    # Adjust feature values for future prediction (e.g., increase population density over time)
    year_offset = year - 2024
    X_pred['PopDnsty'] *= (1 + 0.02 * year_offset)  # Assuming 2% annual growth
    X_pred['Prcpittn'] *= (1 + 0.01 * year_offset)  # Assuming 1% change in precipitation

    for feature in all_features:
        if feature not in selected_features:
            X_pred[feature] = X_pred[feature].mean()

    X_pred.fillna(X_pred.mean(), inplace=True)
    X_pred_scaled = scaler.transform(X_pred)
    predictions = trained_model.predict(X_pred_scaled)
    prediction_data['pred_cases'] = predictions.astype(int)

    center = prediction_data.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=8)

    Choropleth(
        geo_data=prediction_data,
        data=prediction_data,
        columns=['ward_name', 'pred_cases'],
        key_on='feature.properties.ward_name',
        fill_color='YlOrRd',
        fill_opacity=1.0,
        line_opacity=0.1,
        legend_name=f'Predicted Cases ({year})'
    ).add_to(m)

    for _, row in prediction_data.iterrows():
        folium.GeoJson(
            row.geometry,
            tooltip=folium.Tooltip(
                f"Predicted Cases ({year}): {row['pred_cases']}<br>"
                f"Ward: {row['ward_name']}<br>"
                f"LGA: {row['lga_name']}"
            ),
        ).add_to(m)

    return m._repr_html_()

@app.route('/')
def index():
    initial_year = 2025
    map_html = update_map(selected_features, initial_year)
    return render_template('index.html', map_html=map_html, all_features=display_features, year=initial_year)

@app.route('/update', methods=['POST'])
def update():
    global selected_features
    selected_features = request.json['selected_features']
    selected_year = request.json['year']
    map_html = update_map(selected_features, selected_year)
    return jsonify(map_html=map_html)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)
