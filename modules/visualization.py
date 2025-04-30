import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.io as pio

class Visualizer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        self.charts_dir = os.path.join(self.output_dir, "charts")
        self.cards_dir = os.path.join(self.output_dir, "location_cards")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.cards_dir, exist_ok=True)
    
    def plot_poverty_map(self, df):
        """Generate an interactive world map showing poverty scores by location"""
        print("Generating world poverty map...")
        
        # Create figure with mapbox
        fig = px.scatter_mapbox(
            df, 
            lat="latitude", 
            lon="longitude", 
            color="poverty_score",
            size="poverty_score",
            color_continuous_scale=px.colors.sequential.Reds,
            range_color=[0, 100],
            size_max=15,
            zoom=1,
            hover_name="city",
            hover_data=["country", "poverty_score", "poverty_level"],
            title="Global Poverty Index Map"
        )
        
        # Use open-street-map for free map tiles
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=800
        )
        
        # Save as HTML for interactivity
        output_path = os.path.join(self.charts_dir, "poverty_map.html")
        fig.write_html(output_path)
        
        # Also save as static image for reports
        static_path = os.path.join(self.charts_dir, "poverty_map.png")
        pio.write_image(fig, static_path, width=1200, height=800)
        
        print(f"World poverty map saved to {output_path}")
        return output_path
    
    def plot_poverty_distribution(self, df):
        """Generate poverty score distribution visualizations"""
        print("Generating poverty distribution charts...")
        
        # Set up the matplotlib figure
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Histogram of poverty scores
        plt.subplot(2, 2, 1)
        sns.histplot(df['poverty_score'], kde=True, bins=15)
        plt.title('Distribution of Poverty Scores')
        plt.xlabel('Poverty Score')
        plt.ylabel('Frequency')
        
        # Plot 2: Boxplots by region
        plt.subplot(2, 2, 2)
        sns.boxplot(x='region', y='poverty_score', data=df)
        plt.title('Poverty Scores by Region')
        plt.xticks(rotation=45)
        plt.xlabel('Region')
        plt.ylabel('Poverty Score')
        
        # Plot 3: Poverty level counts
        plt.subplot(2, 2, 3)
        sns.countplot(x='poverty_level', data=df, palette='viridis')
        plt.title('Count of Locations by Poverty Level')
        plt.xlabel('Poverty Level')
        plt.ylabel('Count')
        
        # Plot 4: Top 10 countries with highest average poverty
        top_countries = df.groupby('country')['poverty_score'].mean().sort_values(ascending=False).head(10)
        plt.subplot(2, 2, 4)
        sns.barplot(x=top_countries.values, y=top_countries.index, palette='rocket')
        plt.title('Top 10 Countries by Average Poverty Score')
        plt.xlabel('Average Poverty Score')
        plt.ylabel('Country')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.charts_dir, "poverty_distribution.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Poverty distribution charts saved to {output_path}")
        return output_path
    
    def create_location_card(self, city, country, poverty_score, image_data):
        """Create visual card for a specific location"""
        # Convert numpy array to PIL image if necessary
        if isinstance(image_data, np.ndarray):
            satellite_image = Image.fromarray(image_data.astype('uint8'))
        else:
            satellite_image = image_data
        
        # Resize the satellite image
        satellite_image = satellite_image.resize((600, 400))
        
        # Create a new image with space for text
        card = Image.new('RGB', (600, 600), color=(245, 245, 245))
        card.paste(satellite_image, (0, 0))
        
        # Load a font
        try:
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_text = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # Get drawing context
        draw = ImageDraw.Draw(card)
        
        # Determine poverty level and color
        if poverty_score >= 70:
            poverty_level = "High Poverty"
            color = (200, 0, 0)  # Red for high poverty
        elif poverty_score >= 30:
            poverty_level = "Moderate Poverty"
            color = (255, 165, 0)  # Orange for moderate poverty
        else:
            poverty_level = "Low Poverty"
            color = (0, 128, 0)  # Green for low poverty
        
        # Add location information
        draw.text((20, 420), f"{city}, {country}", fill=(0, 0, 0), font=font_title)
        
        # Add poverty score with colored background
        draw.rectangle([(20, 460), (580, 500)], fill=color)
        draw.text((30, 470), f"Poverty Score: {poverty_score:.1f}/100 - {poverty_level}", 
                 fill=(255, 255, 255), font=font_text)
        
        # Add some additional info
        indicators = [
            f"Building Density: {'High' if poverty_score > 50 else 'Medium' if poverty_score > 25 else 'Low'}",
            f"Infrastructure: {'Poor' if poverty_score > 60 else 'Adequate' if poverty_score > 30 else 'Good'}",
            f"Economic Activity: {'Low' if poverty_score > 70 else 'Moderate' if poverty_score > 40 else 'High'}"
        ]
        
        y_pos = 510
        for indicator in indicators:
            draw.text((20, y_pos), indicator, fill=(0, 0, 0), font=font_text)
            y_pos += 25
        
        # Save the card
        safe_city_name = "".join(c if c.isalnum() else "_" for c in city)
        output_path = os.path.join(self.cards_dir, f"{safe_city_name}_{country}.png")
        card.save(output_path)
        
        print(f"Location card for {city}, {country} saved to {output_path}")
        return output_path
    
    def create_comparison_visualization(self, locations_data):
        """Create a comparison visualization of multiple locations"""
        num_locations = len(locations_data)
        if num_locations == 0:
            return None
        
        # Create figure with appropriate size
        fig, axes = plt.subplots(num_locations, 2, figsize=(12, 5 * num_locations))
        
        # Handle single location case
        if num_locations == 1:
            axes = np.array([axes])
        
        for i, location in enumerate(locations_data):
            # Get location info
            city = location.get('city', 'Unknown')
            country = location.get('country', 'Unknown')
            poverty_score = location.get('poverty_score', 0)
            satellite_img = location.get('image')
            
            # Plot satellite image
            ax_img = axes[i, 0]
            ax_img.imshow(satellite_img)
            ax_img.set_title(f"{city}, {country}")
            ax_img.axis('off')
            
            # Plot poverty indicators
            ax_indicators = axes[i, 1]
            
            # Generate some mock indicators based on poverty score
            indicators = {
                'Building Quality': max(0, 100 - poverty_score * 1.2),
                'Infrastructure': max(0, 100 - poverty_score * 1.1),
                'Economic Activity': max(0, 100 - poverty_score * 0.9),
                'Public Services': max(0, 100 - poverty_score * 1.0),
                'Education Access': max(0, 100 - poverty_score * 0.8)
            }
            
            # Plot horizontal bar chart
            y_pos = range(len(indicators))
            indicator_vals = list(indicators.values())
            indicator_names = list(indicators.keys())
            
            bars = ax_indicators.barh(y_pos, indicator_vals, align='center')
            
            # Color bars based on values
            for j, bar in enumerate(bars):
                if indicator_vals[j] < 33:
                    bar.set_color('r')
                elif indicator_vals[j] < 66:
                    bar.set_color('orange')
                else:
                    bar.set_color('g')
            
            ax_indicators.set_yticks(y_pos)
            ax_indicators.set_yticklabels(indicator_names)
            ax_indicators.invert_yaxis()  # Labels read top-to-bottom
            ax_indicators.set_xlabel('Score (higher is better)')
            ax_indicators.set_title(f'Poverty Indicators - Score: {poverty_score:.1f}/100')
            ax_indicators.set_xlim(0, 100)
            
            # Add a text label for poverty level
            if poverty_score >= 70:
                level_text = "HIGH POVERTY"
                level_color = 'red'
            elif poverty_score >= 30:
                level_text = "MODERATE POVERTY"
                level_color = 'orange'
            else:
                level_text = "LOW POVERTY"
                level_color = 'green'
                
            ax_indicators.text(50, -0.5, level_text, ha='center', va='center', 
                               fontsize=12, fontweight='bold', color=level_color)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.charts_dir, "location_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Comparison visualization saved to {output_path}")
        return output_path




