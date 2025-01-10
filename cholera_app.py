import geopandas as gpd
import joblib
from flask import Flask, render_template, jsonify, request
import folium
from folium import Choropleth
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Paths to model, scaler, and shapefile
model_path = r"C:\Users\victor.idakwo\Documents\ehealth Africa\TASK\Cholera-Project-main\New-Cholera-Project\Yobe\Yobe state\models\random-forest-model.joblib"
scaler_path = r"C:\Users\victor.idakwo\Documents\ehealth Africa\TASK\Cholera-Project-main\New-Cholera-Project\Yobe\Yobe state\models\scaler_rf.joblib"
data_path = r"C:\Users\victor.idakwo\Documents\ehealth Africa\TASK\Cholera-Project-main\New-Cholera-Project\Yobe\Yobe state\Population_Cholera.shp"

# Load the trained model and scaler
trained_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the shapefile (GeoDataFrame)
prediction_data = gpd.read_file(data_path)

# Ensure CRS is set to EPSG:4326 (WGS84)
if prediction_data.crs.to_epsg() != 4326:
    prediction_data = prediction_data.to_crs(epsg=4326)

# Features for prediction
all_features = ['Aspect', 'Elevatn', 'builtupr', 'LST', 'LULCC', 'NDVI', 'NDWI', 'PopDnsty', 'Poverty', 'Prcpittn', 'Slope', 'rwi']
selected_features = all_features.copy()  # Initially, all features are selected

# Function to generate the map based on selected features
def update_map(selected_features):
    # Make a copy of the prediction data to avoid modifying the original
    X_pred = prediction_data[all_features].copy()

    # Set values to 0 for the features that are not selected
    for feature in all_features:
        if feature not in selected_features:
            X_pred[feature] = 0  # Set unselected features to zero

    # Check and handle missing values
    if X_pred.isnull().any().any():
        X_pred = X_pred.fillna(X_pred.mean())

    # Scale the prediction data
    X_pred_scaled = scaler.transform(X_pred)

    # Make predictions
    predictions = trained_model.predict(X_pred_scaled)
    prediction_data['pred_cases'] = predictions.astype(int)

    # Calculate the center of the map based on centroids
    center = prediction_data.geometry.centroid.unary_union.centroid

    # Create a Folium map dynamically centered
    m = folium.Map(location=[center.y, center.x], zoom_start=8)

    # Add choropleth layer for visualizing predicted cases
    Choropleth(
        geo_data=prediction_data,
        data=prediction_data,
        columns=['ward_name', 'pred_cases'],
        key_on='feature.properties.ward_name',
        fill_color='YlOrRd',
        fill_opacity=1.0,
        line_opacity=0.1,
        legend_name='Predicted Cases'
    ).add_to(m)

    # Add a hover popup (tooltip) for each ward
    for _, row in prediction_data.iterrows():
        folium.GeoJson(
            row.geometry,
            tooltip=folium.Tooltip(
                f"Predicted Cases: {row['pred_cases']}<br>"
                f"Ward: {row['ward_name']}<br>"
                f"LGA: {row['lga_name']}"
            ),
        ).add_to(m)

    # Return the map as HTML
    return m._repr_html_()

@app.route('/')
def index():
    # Initially, load the map with all features selected
    map_html = update_map(selected_features)
    return render_template('index.html', map_html=map_html, all_features=all_features)

@app.route('/update', methods=['POST'])
def update():
    # Get selected features from the request
    selected_features = request.json['selected_features']
    # Generate the updated map
    map_html = update_map(selected_features)
    return jsonify(map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
