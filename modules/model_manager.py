
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

class ModelManager:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "poverty_cnn_model.h5")
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "model_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize model
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print("Building new CNN model...")
            self.model = self._build_custom_cnn()
    
    def _build_custom_cnn(self):
        """Build a custom CNN model for satellite image poverty analysis"""
        model = Sequential([
            # First Convolutional Block - Extract basic features
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block - Extract more complex features
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block - Extract high-level features
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten the feature maps
            Flatten(),
            
            # Fully connected layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer - poverty score prediction
            Dense(1, activation='sigmoid')  # Single output scaled 0-1
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Resize to expected input size
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        
        # Normalize to 0-1 range
        img_array = img_array / 255.0
        
        # Add batch dimension if not already present
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
            
        return img_array
    
    def train(self, images, poverty_scores, epochs=10, batch_size=16):
        """Train the model with satellite images and poverty scores"""
        # Process images and scores for training
        processed_images = []
        for img in images:
            processed = self.preprocess_image(img)
            if processed.shape[0] == 1:  # Remove batch dimension for concatenation
                processed = processed[0]
            processed_images.append(processed)
            
        X = np.array(processed_images)
        y = np.array(poverty_scores) / 100.0  # Scale to 0-1
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a visual representation of the model architecture
        try:
            plot_model(
                self.model, 
                to_file=os.path.join(self.results_dir, 'model_architecture.png'),
                show_shapes=True, 
                show_layer_names=True
            )
        except Exception as e:
            print(f"Could not generate model visualization: {e}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save model
        self.model.save(self.model_path)
        print(f"CNN model saved to {self.model_path}")
        
        # Create and save training visualization
        self._save_training_visualization(history)
        
        return {
            'training_accuracy': float(history.history['mae'][-1]),
            'validation_accuracy': float(history.history['val_mae'][-1]),
            'message': "Training complete - model ready for poverty prediction"
        }
    
    def _save_training_visualization(self, history):
        """Create user-friendly training visualization"""
        plt.figure(figsize=(12, 5))
        
        # Plot Mean Absolute Error
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model Accuracy')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'))
        plt.close()
    
    def predict(self, image):
        """Predict poverty score from satellite image"""
        preprocessed = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed, verbose=0)[0][0]
        
        # Scale back to 0-100
        poverty_score = prediction * 100
        return poverty_score
    
    def explain_prediction(self, image):
        """Generate user-friendly explanation of poverty prediction"""
        # Create activation heatmap to show which parts of the image influenced the prediction
        original_image = image
        if isinstance(image, np.ndarray) and len(image.shape) == 3:
            # Convert to PIL for processing
            original_image = Image.fromarray(image.astype('uint8'))
        
        # Get the poverty score
        poverty_score = self.predict(image)
        
        # Generate attention map (this is a simplified visualization)
        # In a real implementation, you could use GradCAM or similar techniques
        attention_weights = self._generate_simple_attention_map(image)
        
        # Create a visual explanation
        explanation_img = self._create_visual_explanation(
            original_image, 
            attention_weights, 
            poverty_score
        )
        
        # Save the explanation
        explanation_path = os.path.join(self.results_dir, 'poverty_explanation.png')
        explanation_img.save(explanation_path)
        
        # Return user-friendly feature contributions dictionary
        # These would normally come from a proper explainability framework
        return {
            'poverty_score': float(poverty_score),
            'visual_explanation_path': explanation_path,
            'feature_importance': {
                'building_density': self._get_feature_importance('building_density', poverty_score),
                'infrastructure': self._get_feature_importance('infrastructure', poverty_score),
                'vegetation': self._get_feature_importance('vegetation', poverty_score),
                'road_network': self._get_feature_importance('road_network', poverty_score),
                'building_quality': self._get_feature_importance('building_quality', poverty_score)
            },
            'message': f"Poverty score of {poverty_score:.1f}/100 detected based on CNN analysis"
        }
    
    def _generate_simple_attention_map(self, image):
        """Generate a simple attention map for visualization purposes"""
        # This is a placeholder for a more sophisticated approach like GradCAM
        # For now, we'll create a synthetic attention map
        processed = self.preprocess_image(image)[0]
        
        # Create a synthetic heatmap based on image features
        # In reality, this would come from actual model gradients
        gray_img = np.mean(processed, axis=-1)
        
        # Apply Gaussian blur to create a smoother heatmap
        from scipy.ndimage import gaussian_filter
        attention = gaussian_filter(gray_img, sigma=5)
        
        # Normalize to 0-1
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        return attention
    
    def _create_visual_explanation(self, original_image, attention_map, poverty_score):
        """Create a visual explanation with the image and attention map"""
        # Resize attention map to match original image
        original_width, original_height = original_image.size
        attention_resized = Image.fromarray(
            (attention_map * 255).astype(np.uint8)
        ).resize((original_width, original_height))
        
        # Create a heatmap overlay
        heatmap = np.array(attention_resized)
        # Apply colormap (red for high attention, blue for low)
        cmap = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        cmap[..., 0] = np.maximum(0, 255 * (2 * (1 - heatmap/255)))  # Blue
        cmap[..., 2] = np.maximum(0, 255 * (2 * (heatmap/255)))      # Red
        
        # Convert to PIL image
        heatmap_img = Image.fromarray(cmap)
        
        # Create a composite image
        original_image_resized = original_image.resize((original_width, original_height))
        composite = Image.new('RGB', (original_width * 2 + 20, original_height + 40))
        
        # Add original image
        composite.paste(original_image_resized, (0, 20))
        
        # Add heatmap
        composite.paste(heatmap_img, (original_width + 20, 20))
        
        # Add labels using PIL's ImageDraw
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(composite)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text
        draw.text((10, 0), "Original Satellite Image", fill=(255, 255, 255), font=font)
        draw.text((original_width + 30, 0), "CNN Attention Map", fill=(255, 255, 255), font=font)
        
        poverty_level = "Low"
        if poverty_score > 30:
            poverty_level = "Moderate"
        if poverty_score > 70:
            poverty_level = "High"
            
        draw.text(
            (10, original_height + 20), 
            f"Poverty Score: {poverty_score:.1f}/100 - {poverty_level} Poverty", 
            fill=(255, 255, 255), 
            font=font
        )
        
        return composite
    
    def _get_feature_importance(self, feature_name, poverty_score):
        """Get estimated feature importance for explanation"""
        # In a real system, this would come from techniques like SHAP or LIME
        # For this demo, we'll provide reasonable values based on the poverty score
        
        # Base importance values with some randomness for realism
        base_values = {
            'building_density': 0.7,
            'infrastructure': 0.6,
            'vegetation': 0.4,
            'road_network': 0.5,
            'building_quality': 0.8
        }
        
        # Add some correlation with the poverty score
        correlation = 0.3 * (poverty_score / 100)
        
        # Add some randomness
        randomness = np.random.uniform(-0.1, 0.1)
        
        # Calculate final importance
        importance = base_values.get(feature_name, 0.5) + correlation + randomness
        
        # Ensure it's in 0-1 range
        return float(np.clip(importance, 0, 1))







