import os
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

class DataManager:
    def __init__(self, satellite_imagery):
        self.satellite = satellite_imagery
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset_path = os.path.join(self.data_dir, "poverty_dataset.csv")
        
        # Regions and countries for simulated data
        self.regions = {
            'Africa': ['Nigeria', 'Kenya', 'South Africa', 'Ethiopia', 'Egypt', 'Ghana'],
            'Asia': ['India', 'China', 'Japan', 'Bangladesh', 'Philippines', 'Vietnam'],
            'Europe': ['Germany', 'France', 'UK', 'Italy', 'Spain', 'Poland'],
            'North America': ['USA', 'Canada', 'Mexico', 'Cuba', 'Guatemala', 'Haiti'],
            'South America': ['Brazil', 'Argentina', 'Colombia', 'Peru', 'Chile', 'Venezuela'],
            'Oceania': ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands']
        }
        
        # Regional bias for simulated poverty scores
        self.regional_poverty_bias = {
            'Africa': 60,
            'Asia': 45,
            'Europe': 20,
            'North America': 25,
            'South America': 40,
            'Oceania': 30
        }
        
        # Initialize geocoder for reverse lookup
        self.geocoder = Nominatim(user_agent="poverty_prediction_system")
        
    def load_dataset(self):
        """Load existing poverty dataset if available"""
        if os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                print(f"Loaded dataset with {len(df)} locations")
                return df
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return None
        else:
            print("No existing dataset found")
            return None
    
    def save_dataset(self, df):
        """Save poverty dataset to CSV"""
        df.to_csv(self.dataset_path, index=False)
        print(f"Dataset saved with {len(df)} locations")
    
    def generate_dataset(self, num_locations=100):
        """Generate simulated poverty dataset"""
        data = []
        
        print(f"Generating {num_locations} locations...")
        for i in range(num_locations):
            # Bias selection toward lower-middle income countries
            region = random.choices(
                list(self.regions.keys()), 
                weights=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05],
                k=1
            )[0]
            
            # Select country from region
            country = random.choice(self.regions[region])
            
            # Get a random satellite image 
            lat, lon, image = self.satellite.get_random_image()
            
            # Get location info
            location_info = self._get_location_info(lat, lon)
            city = location_info.get('city', 'Unknown')
            
            # Generate simulated poverty score with regional bias
            base_score = self.regional_poverty_bias.get(region, 50)
            
            # Add variability
            poverty_score = np.clip(
                np.random.normal(base_score, 15),
                0, 100
            )
            
            # Create record
            record = {
                'id': i + 1,
                'city': city,
                'country': country,
                'region': region,
                'latitude': lat,
                'longitude': lon,
                'poverty_score': round(poverty_score, 1),
                'poverty_level': self._classify_poverty(poverty_score),
                'date_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Save the image to cache if needed (the satellite module does this already)
            
            data.append(record)
            print(f"Generated location {i+1}/{num_locations}: {city}, {country}", end='\r')
        
        print("\nDataset generation complete")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save dataset
        self.save_dataset(df)
        
        return df
    
    def _get_location_info(self, lat, lon):
        """Get location info from coordinates"""
        result = {
            'city': 'Unknown City',
            'country': 'Unknown Country'
        }
        
        # Try to get actual location data
        try:
            location = self.geocoder.reverse((lat, lon), language='en', timeout=5)
            if location:
                address = location.raw.get('address', {})
                
                # Extract city - try different fields as different regions use different formats
                for city_field in ['city', 'town', 'village', 'county']:
                    if city_field in address:
                        result['city'] = address[city_field]
                        break
                
                # Extract country
                if 'country' in address:
                    result['country'] = address['country']
            
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Geocoder error: {e}")
        
        # Generate plausible names if we couldn't get real data
        if result['city'] == 'Unknown City':
            # First syllables
            first = ['New', 'San', 'Los', 'East', 'West', 'North', 'South', 
                     'Port', 'Fort', 'Mount', 'Lake', 'Bay']
            # Second syllables
            second = ['York', 'Francisco', 'Angeles', 'Vegas', 'Ridge', 'View', 
                      'Haven', 'Springs', 'Wood', 'Town', 'City', 'Vale']
            
            result['city'] = f"{random.choice(first)} {random.choice(second)}"
        
        if result['country'] == 'Unknown Country':
            # Random country from any region
            all_countries = [c for countries in self.regions.values() for c in countries]
            result['country'] = random.choice(all_countries)
        
        return result
    
    def _classify_poverty(self, score):
        """Classify poverty score into categories"""
        if score >= 70:
            return 'High'
        elif score >= 30:
            return 'Moderate'
        else:
            return 'Low'
    
    def add_location(self, lat, lon, poverty_score=None):
        """Add a new location to the dataset"""
        # Load current dataset
        df = self.load_dataset()
        if df is None:
            df = pd.DataFrame(columns=[
                'id', 'city', 'country', 'region', 'latitude', 'longitude', 
                'poverty_score', 'poverty_level', 'date_updated'
            ])
        
        # Get location info
        location_info = self._get_location_info(lat, lon)
        
        # If poverty score not provided, generate it
        if poverty_score is None:
            # Get region
            region = None
            for r, countries in self.regions.items():
                if location_info['country'] in countries:
                    region = r
                    break
            
            if region is None:
                region = random.choice(list(self.regions.keys()))
            
            # Generate score with regional bias
            base_score = self.regional_poverty_bias.get(region, 50)
            poverty_score = np.clip(np.random.normal(base_score, 15), 0, 100)
        
        # Create new record
        new_id = 1 if len(df) == 0 else df['id'].max() + 1
        new_record = {
            'id': new_id,
            'city': location_info['city'],
            'country': location_info['country'],
            'region': next((r for r, countries in self.regions.items() 
                           if location_info['country'] in countries), 'Other'),
            'latitude': lat,
            'longitude': lon,
            'poverty_score': round(float(poverty_score), 1),
            'poverty_level': self._classify_poverty(poverty_score),
            'date_updated': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Add to dataframe
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save updated dataset
        self.save_dataset(df)
        
        return new_record
    
    def get_location_data(self, lat, lon):
        """Get data for a specific location"""
        # First check if location exists in dataset
        df = self.load_dataset()
        if df is not None:
            # Find closest location within reasonable distance (0.1 degree ~ 11km at equator)
            close_locations = df[
                (abs(df['latitude'] - lat) < 0.1) & 
                (abs(df['longitude'] - lon) < 0.1)
            ]
            
            if len(close_locations) > 0:
                # Return the closest one
                closest = close_locations.iloc[0].to_dict()
                return closest
        
        # If not in dataset, fetch image and location info
        try:
            image = self.satellite.get_location_image(lat, lon)
            location_info = self._get_location_info(lat, lon)
            
            # Return without adding to dataset
            return {
                'city': location_info['city'],
                'country': location_info['country'],
                'latitude': lat,
                'longitude': lon,
                'image': image
            }
        except Exception as e:
            print(f"Error getting location data: {e}")
            return None
    
    def get_locations_by_poverty_level(self, level, limit=10):
        """Get locations by poverty level category"""
        df = self.load_dataset()
        if df is None:
            return []
        
        filtered = df[df['poverty_level'] == level].head(limit)
        return filtered.to_dict('records')
    
    def export_data(self, format='json'):
        """Export data in specified format"""
        df = self.load_dataset()
        if df is None:
            return None
        
        if format.lower() == 'json':
            output_path = os.path.join(self.data_dir, "poverty_data.json")
            df.to_json(output_path, orient='records', indent=2)
            return output_path
        elif format.lower() == 'csv':
            return self.dataset_path
        else:
            raise ValueError(f"Unsupported export format: {format}")

