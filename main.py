import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
import numpy as np
import streamlit as st

class WaterResourceManager:
    def __init__(self, water_usage_data=None, farming_areas_data=None):
        """
        Initializes the WaterResourceManager with optional data.
        """
        self.water_usage_data = water_usage_data
        self.farming_areas_data = farming_areas_data
        self.model = None
        self.scaler = None
        self.trained = False
        self.data = None  # Initialize self.data here
        self.data_loaded = False # Initialize data load checker
        self.preprocessed = False # Track if pre-processing has occurred

    def load_sample_data(self, num_farms=50):
        """
        Generates sample data for water usage and farming areas.
        """
        np.random.seed(42)  # For reproducibility

        # Water Usage Data
        farm_ids = [f"Farm_{i+1}" for i in range(num_farms)]
        water_usage = np.random.randint(500, 5000, num_farms)  # Liters per period
        crop_types = np.random.choice(["Wheat", "Corn", "Soybean", "Rice", "Alfalfa"], num_farms)
        soil_types = np.random.choice(["Sandy", "Clay", "Loam"], num_farms)
        rainfall = np.random.randint(10, 100, num_farms)  # mm per period
        temperature = np.random.uniform(10, 35, num_farms)  # Celsius

        self.water_usage_data = pd.DataFrame({
            "FarmID": farm_ids,
            "WaterUsage": water_usage,
            "CropType": crop_types,
            "SoilType": soil_types,
            "Rainfall": rainfall,
            "Temperature": temperature
        })

        # Farming Areas Data (Sample Coordinates)
        longitudes = np.random.uniform(-125, -65, num_farms)
        latitudes = np.random.uniform(25, 50, num_farms)
        geometry = gpd.points_from_xy(longitudes, latitudes)

        self.farming_areas_data = gpd.GeoDataFrame({
            "FarmID": farm_ids,
            "geometry": geometry,
            "Area": np.random.randint(10, 100, num_farms)
        }, crs="EPSG:4326")

        self.data_loaded = True
        st.write("Loaded Sample data")
        self.preprocessed = False

    def load_data_from_files(self, water_usage_file, farming_areas_file):
        """
        Loads data from CSV or GeoJSON files.
        """
        try:
            self.water_usage_data = pd.read_csv(water_usage_file)
            self.farming_areas_data = gpd.read_file(farming_areas_file)
            self.data_loaded = True
            st.write(f"Loaded Water usage data from {water_usage_file}")
            st.write(f"Loaded Farming areas data from {farming_areas_file}")
            self.preprocessed = False
        except FileNotFoundError:
            st.error("Error: One or both of the specified files were not found.")
        except Exception as e:
            st.error(f"An error occurred while loading data from files: {e}")

    def load_user_data(self, water_data, area_data):
        """
        Loads user data from provided DataFrames.
        """
        try:
          self.water_usage_data = pd.DataFrame(water_data)
          area_df = pd.DataFrame(area_data)
          geometry = gpd.points_from_xy(area_df.longitude, area_df.latitude)
          self.farming_areas_data = gpd.GeoDataFrame(area_df, geometry=geometry, crs="EPSG:4326")
          self.data_loaded = True
          st.success("User data loaded.")
          self.preprocessed = False
        except Exception as e:
          st.error(f"An error occurred while loading user data: {e}")

    def preprocess_data(self):
        if not self.data_loaded: # Check for data_loaded
            st.error("Error: No data has been loaded yet. Load data first.")
            return
        
        if self.preprocessed:
            st.warning("Data has already been preprocessed.")
            return

        if not isinstance(self.water_usage_data, pd.DataFrame):
            st.error("Error: Water Usage data not loaded.")
            return

        if not isinstance(self.farming_areas_data, gpd.GeoDataFrame):
            st.error("Error: Farming Areas data not loaded.")
            return

        st.write("Preprocessing data")

        # One-hot encode categorical features for the model
        self.water_usage_data = pd.get_dummies(self.water_usage_data, columns=['CropType', 'SoilType'])

        # Merge dataframes
        self.data = pd.merge(self.water_usage_data, self.farming_areas_data, on='FarmID', how='left')

        # Check for null values
        if self.data.isnull().values.any():
            st.warning("Warning: Data contains Null values")
            self.data = self.data.dropna()
            st.write("Null values dropped")

        self.data = self.data.drop(columns=['geometry'])
        self.trained = False
        self.preprocessed = True

    def train_model(self):
        """
        Trains a simple linear regression model to predict water usage.
        """
        if not self.preprocessed:
            st.error("Error: Data must be preprocessed before training the model.")
            return
        if self.data is None: # Check if data is available
            st.error("Error: Data must be preprocessed before training the model.")
            return
        if not self.trained:
            st.write("Training the model")
            X = self.data.drop(["FarmID", "WaterUsage"], axis=1)
            y = self.data["WaterUsage"]

            # Scale numerical values
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            self.trained = True

            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Model MSE: {mse}")
        else:
            st.write("Model is already trained")

    def predict_water_usage(self, new_data):
        if not self.trained:
            st.error("Error: Model not trained")
            return

        if self.data is None:
            st.error("Error: Data is not preprocessed")
            return

        # Ensure the new_data is a DataFrame
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame([new_data])

        # One-hot encode categorical features in new_data
        new_data = pd.get_dummies(new_data, columns=['CropType', 'SoilType'])

        # Align columns to match the training data columns
        train_cols = self.data.drop(["FarmID", "WaterUsage"], axis=1).columns
        for col in train_cols:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[train_cols]  # Reorder to match training order

        # Scale numerical values
        new_data_scaled = self.scaler.transform(new_data)

        predictions = self.model.predict(new_data_scaled)
        return predictions

    def visualize_water_usage(self):
        """
        Visualizes water usage data using various plots.
        """
        if not isinstance(self.water_usage_data, pd.DataFrame):
            st.error("Error: Data not loaded.")
            return

        # Basic histogram of water usage
        st.write("## Distribution of Water Usage")
        fig1, ax1 = plt.subplots()
        sns.histplot(self.water_usage_data["WaterUsage"], kde=True, ax=ax1)
        plt.xlabel("Water Usage (Liters)")
        plt.ylabel("Frequency")
        st.pyplot(fig1)

        # Water usage by crop type
        st.write("## Water Usage by Crop Type")
        fig2, ax2 = plt.subplots()
        sns.barplot(x="CropType", y="WaterUsage", data=self.water_usage_data, ax=ax2)
        plt.xlabel("Crop Type")
        plt.ylabel("Water Usage (Liters)")
        st.pyplot(fig2)

        # Water usage vs. rainfall
        st.write("## Water Usage vs. Rainfall")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x="Rainfall", y="WaterUsage", data=self.water_usage_data, ax=ax3)
        plt.xlabel("Rainfall (mm)")
        plt.ylabel("Water Usage (Liters)")
        st.pyplot(fig3)

        # Water usage vs. temperature
        st.write("## Water Usage vs. Temperature")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x="Temperature", y="WaterUsage", data=self.water_usage_data, ax=ax4)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Water Usage (Liters)")
        st.pyplot(fig4)

    def visualize_spatial_data(self):
        if not isinstance(self.farming_areas_data, gpd.GeoDataFrame):
            st.error("Error: Farming Areas Data not loaded.")
            return

        if not isinstance(self.water_usage_data, pd.DataFrame):
            st.error("Error: Water Usage Data not loaded.")
            return

        # Merge water usage data with spatial data
        merged_data = self.farming_areas_data.merge(self.water_usage_data, on="FarmID")

        # Create a Folium map
        m = folium.Map(location=[merged_data.geometry.y.mean(), merged_data.geometry.x.mean()], zoom_start=6)

        # Create a HeatMap for water usage
        heat_data = [[row['geometry'].y, row['geometry'].x, row['WaterUsage']] for index, row in merged_data.iterrows()]
        HeatMap(heat_data, radius=25).add_to(m)

        # Add circles to the map
        for index, row in merged_data.iterrows():
            folium.CircleMarker(
                location=[row['geometry'].y, row['geometry'].x],
                radius=row['Area'] / 5,  # Adjust radius based on area
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup=f"Farm: {row['FarmID']}<br>Water Usage: {row['WaterUsage']} Liters"
            ).add_to(m)

        st.write("## Spatial Data Visualization")
        st.write("Below is the map that visualizes the farming areas and water usage.")
        st.components.v1.html(m._repr_html_(), width=800, height=600)

def main():
    st.title("Water Resource Management App")

    manager = WaterResourceManager()

    # Choose data source in app
    data_source = st.sidebar.selectbox(
        "Select Data Source", ["Sample", "Files", "User"]
    )

    if data_source == "Sample":
        if st.sidebar.button("Load Sample Data"):
            manager.load_sample_data(num_farms=50)
    elif data_source == "Files":
        water_usage_file = st.sidebar.file_uploader("Upload Water Usage CSV", type="csv")
        farming_areas_file = st.sidebar.file_uploader("Upload Farming Areas GeoJSON", type="json")
        if st.sidebar.button("Load Data from Files"):
            if water_usage_file and farming_areas_file:
                # Save uploaded files temporarily for processing
                with open("temp_water_usage.csv", "wb") as f:
                    f.write(water_usage_file.getvalue())
                with open("temp_farming_areas.json", "wb") as f:
                    f.write(farming_areas_file.getvalue())
                manager.load_data_from_files("temp_water_usage.csv", "temp_farming_areas.json")
            else:
                st.error("Please upload both files.")

    elif data_source == "User":
        st.header("Enter Water Usage Data:")
        num_farms = st.number_input("Number of Farms", min_value=1, value=1, step=1)
        water_data = []
        for i in range(num_farms):
            st.subheader(f"Farm {i+1}")
            farm_id = st.text_input(f"FarmID {i+1}")
            water_usage = st.number_input(f"Water Usage (Liters) {i+1}", min_value=0, value=1000, step=1)
            crop_type = st.selectbox(f"Crop Type {i+1}", ["Wheat", "Corn", "Soybean", "Rice", "Alfalfa"])
            soil_type = st.selectbox(f"Soil Type {i+1}", ["Sandy", "Clay", "Loam"])
            rainfall = st.number_input(f"Rainfall (mm) {i+1}", min_value=0, value=50, step=1)
            temperature = st.number_input(f"Temperature (°C) {i+1}", min_value=-20.0, max_value=50.0, value=25.0, step=0.1)

            water_data.append({
                "FarmID": farm_id,
                "WaterUsage": water_usage,
                "CropType": crop_type,
                "SoilType": soil_type,
                "Rainfall": rainfall,
                "Temperature": temperature
            })

        st.header("Enter Farming Areas Data:")
        area_data = []
        for i in range(num_farms):
            st.subheader(f"Farm {i+1}")
            farm_id = water_data[i]["FarmID"]
            longitude = st.number_input(f"Longitude {i+1}", min_value=-180.0, max_value=180.0, value=-100.0, step=0.1)
            latitude = st.number_input(f"Latitude {i+1}", min_value=-90.0, max_value=90.0, value=40.0, step=0.1)
            area = st.number_input(f"Area {i+1}", min_value=0, value=50, step=1)
            area_data.append({
                "FarmID": farm_id,
                "longitude": longitude,
                "latitude": latitude,
                "Area": area
            })
        if st.button("Load User data"):
          manager.load_user_data(water_data, area_data)

    # Preprocess data button
    if st.button("Preprocess Data"):
        manager.preprocess_data()

    # Train model button
    if st.button("Train Model"):
        manager.train_model()

    # Example prediction
    st.header("Make a Prediction")
    new_farm_data = {
        "Rainfall": st.number_input("New Rainfall (mm)", min_value=0, value=50, step=1),
        "Temperature": st.number_input("New Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.1),
        "CropType": st.selectbox("New Crop Type", ["Wheat", "Corn", "Soybean", "Rice", "Alfalfa"]),
        "SoilType": st.selectbox("New Soil Type", ["Sandy", "Clay", "Loam"]),
        "Area": st.number_input("New Area", min_value=0, value=50, step=1)
    }
    if st.button("Predict Water Usage"):
        predicted_water_usage = manager.predict_water_usage(pd.DataFrame([new_farm_data]))
        st.write(f"Predicted Water Usage: {predicted_water_usage[0]:.2f} Liters")

    # Visualization buttons
    st.header("Visualizations")
    if st.button("Visualize Water Usage Trends"):
        manager.visualize_water_usage()

    if st.button("Visualize Spatial Data"):
        manager.visualize_spatial_data()


if __name__ == "__main__":
    main()