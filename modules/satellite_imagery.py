import requests
import numpy as np
from io import BytesIO
from PIL import Image
import random
import os

class SatelliteImagery:
    def __init__(self, mapbox_token=None):
        self.mapbox_token = mapbox_token or "YOUR_MAPBOX_TOKEN"
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "image_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_random_image(self):
        """Get satellite image from random location"""
        # Generate random coordinates (land-biased)
        lat = random.uniform(-60, 70)  # Avoiding extreme latitudes
        lon = random.uniform(-180, 180)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                img = self.get_location_image(lat, lon)
                return lat, lon, img
            except Exception as e:
                print(f"Error fetching image (attempt {attempt+1}/{max_attempts}): {e}")
                # Generate slightly different coordinates for retry
                lat += random.uniform(-1, 1)
                lon += random.uniform(-1, 1)
                
        # Return placeholder image after all attempts fail
        print("All attempts to fetch image failed, returning placeholder")
        return lat, lon, np.zeros((224, 224, 3), dtype=np.uint8)
        
    def get_location_image(self, lat, lon):
        """Get high-quality map image from Mapbox"""
        # Check cache first
        cache_path = os.path.join(self.cache_dir, f"{lat:.4f}_{lon:.4f}.jpg")
        if os.path.exists(cache_path):
            try:
                img = Image.open(cache_path)
                return np.array(img)
            except Exception:
                # If cache read fails, continue to fetch new image
                pass
        
        if self.mapbox_token == "YOUR_MAPBOX_TOKEN":
            print("Warning: Using placeholder Mapbox token. Returning simulated image.")
            img_array = self._generate_simulated_image(lat, lon)
            
            # Cache the simulated image
            img = Image.fromarray(img_array)
            img.save(cache_path)
            
            return img_array
            
        url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},15,0/800x800?access_token={self.mapbox_token}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Mapbox API error: {response.status_code}")
            
        img = Image.open(BytesIO(response.content))
        
        # Save to cache
        img.save(cache_path)
        
        return np.array(img)
    
    def _generate_simulated_image(self, lat, lon):
        """Generate CNN-friendly simulated satellite image when API is unavailable"""
        # Size for CNN input (224x224x3)
        image_size = 224
        
        # Base colors
        urban_color = np.array([100, 100, 100])  # Gray for urban
        rural_color = np.array([30, 120, 30])    # Green for rural
        water_color = np.array([30, 30, 150])    # Blue for water
        road_color = np.array([180, 180, 180])   # Light gray for roads
        building_color = np.array([130, 110, 90])  # Brown for buildings
        
        # Urbanization probability depends on latitude
        # More urban near equator, more rural at high latitudes
        urban_prob = 1 - (abs(lat) / 90) * 0.8
        
        # Poverty level simulated based on longitude (totally arbitrary)
        # This creates some regional bias for the model to learn
        poverty_factor = (np.sin(lon / 30) + 1) / 2  # 0-1 range
        
        # Generate base image - use float32 to avoid overflow during operations
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)
        
        # Add base terrain
        if random.random() < 0.3:  # 30% chance of water body
            img += np.random.randint(0, 20, (image_size, image_size, 3)).astype(np.float32)
            water_region = np.random.rand(image_size, image_size) < 0.6
            water_noise = np.random.randint(-20, 20, 3).astype(np.float32)
            for i in range(3):
                img_section = img[:, :, i]
                img_section[water_region] = float(water_color[i]) + water_noise[i]
                img[:, :, i] = img_section
        else:
            # Base terrain with some noise
            for i in range(3):
                img[:,:,i] = np.random.randint(
                    rural_color[i] - 30, 
                    rural_color[i] + 30, 
                    (image_size, image_size)
                ).astype(np.float32)
        
        # Add urban elements based on urbanization probability
        if random.random() < urban_prob:
            # Create a city layout with blocks and roads
            block_size = random.randint(15, 40)
            
            # Add grid-like roads
            for i in range(0, image_size, block_size):
                road_width = random.randint(2, 5)
                
                # Horizontal roads
                if i < image_size - road_width:
                    road_noise = np.random.randint(-30, 30, 3).astype(np.float32)
                    for j in range(3):
                        img[i:i+road_width, :, j] = float(road_color[j]) + road_noise[j]
                
                # Vertical roads
                if i < image_size - road_width:
                    road_noise = np.random.randint(-30, 30, 3).astype(np.float32)
                    for j in range(3):
                        img[:, i:i+road_width, j] = float(road_color[j]) + road_noise[j]
            
            # Add buildings in blocks
            for y in range(block_size//2, image_size, block_size):
                for x in range(block_size//2, image_size, block_size):
                    if y >= image_size or x >= image_size:
                        continue
                        
                    # Skip if on a road
                    if (y % block_size < 5) or (x % block_size < 5):
                        continue
                    
                    # Building density based on poverty factor
                    # More poverty = more dense, smaller buildings
                    if random.random() < 0.8:  # 80% chance of building in block
                        if poverty_factor > 0.6:  # High poverty
                            # Many small buildings clustered together
                            for _ in range(random.randint(3, 6)):
                                bx = x + random.randint(-block_size//3, block_size//3)
                                by = y + random.randint(-block_size//3, block_size//3)
                                if bx < 0 or by < 0 or bx >= image_size or by >= image_size:
                                    continue
                                
                                b_size = random.randint(5, 10)
                                if bx + b_size >= image_size or by + b_size >= image_size:
                                    continue
                                    
                                # Poverty areas have more variable, darker buildings
                                b_color = building_color - np.array([40, 40, 40]) + np.random.randint(-30, 10, 3).astype(np.float32)
                                for j in range(3):
                                    img[by:by+b_size, bx:bx+b_size, j] = b_color[j]
                        else:  # Low poverty
                            # Larger, more uniform buildings
                            b_size = random.randint(block_size//3, block_size//2)
                            if x + b_size >= image_size or y + b_size >= image_size:
                                continue
                                
                            # Wealthier areas have lighter, more uniform buildings
                            b_color = building_color + np.array([40, 40, 40]) + np.random.randint(-10, 30, 3).astype(np.float32)
                            for j in range(3):
                                img[y:y+b_size, x:x+b_size, j] = b_color[j]
                            
                            # Add some green space around wealthy buildings
                            if random.random() < 0.4:
                                g_size = random.randint(3, 8)
                                gx = x + random.randint(-5, b_size)
                                gy = y + random.randint(-5, b_size)
                                if 0 <= gx < image_size-g_size and 0 <= gy < image_size-g_size:
                                    g_color = rural_color + np.random.randint(-20, 20, 3).astype(np.float32)
                                    for j in range(3):
                                        img[gy:gy+g_size, gx:gx+g_size, j] = g_color[j]
        else:
            # Rural area - add scattered buildings and more vegetation
            for _ in range(random.randint(10, 30)):
                x = random.randint(0, image_size-20)
                y = random.randint(0, image_size-20)
                b_size = random.randint(5, 15)
                b_color = building_color + np.random.randint(-20, 20, 3).astype(np.float32)
                for j in range(3):
                    img[y:y+b_size, x:x+b_size, j] = b_color[j]
            
            # Add some fields/farmland
            for _ in range(random.randint(3, 8)):
                x = random.randint(0, image_size-50)
                y = random.randint(0, image_size-50)
                field_size = random.randint(30, 50)
                
                if x + field_size >= image_size or y + field_size >= image_size:
                    continue
                    
                # Create field with slightly different green
                field_color = rural_color + np.random.randint(-10, 10, 3).astype(np.float32)
                for j in range(3):
                    img[y:y+field_size, x:x+field_size, j] = field_color[j]
                
                # Add patterns in fields
                pattern = np.random.rand(field_size, field_size) > 0.5
                pattern_color = field_color + np.array([15, 15, 0])
                for i in range(3):
                    field_section = img[y:y+field_size, x:x+field_size, i]
                    field_section[pattern] = pattern_color[i]
                    img[y:y+field_size, x:x+field_size, i] = field_section
        
        # Add some final noise for texture (carefully to avoid overflow)
        noise = np.random.randint(-10, 10, (image_size, image_size, 3)).astype(np.float32)
        img = img + noise
        
        # Clip values to valid uint8 range before converting
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img



