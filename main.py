import os
import argparse
import numpy as np
from modules.data_manager import DataManager
from modules.visualization import Visualizer
from modules.satellite_imagery import SatelliteImagery
from modules.model_manager import ModelManager

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Poverty Prediction System')
    parser.add_argument('--generate', type=int, default=0, 
                        help='Generate new dataset with specified number of locations')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations from dataset')
    parser.add_argument('--train', action='store_true',
                        help='Train model with dataset')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze a specific location by coordinates (format: lat,lon)')
    parser.add_argument('--mapbox-token', type=str, default=None,
                        help='Mapbox API token')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Get Mapbox token from environment if not provided
    mapbox_token = args.mapbox_token or os.environ.get('MAPBOX_TOKEN', 'YOUR_MAPBOX_TOKEN')
    
    # Initialize components
    satellite = SatelliteImagery(mapbox_token)
    data_manager = DataManager(satellite)
    visualizer = Visualizer(data_manager)
    model_manager = ModelManager()
    
    # Generate or load dataset
    if args.generate > 0:
        print(f"Generating dataset with {args.generate} locations...")
        df = data_manager.generate_dataset(args.generate)
        print(f"Dataset generated with {len(df)} locations")
    else:
        df = data_manager.load_dataset()
        if df is None:
            print("No dataset found. Generating sample dataset with 10 locations...")
            df = data_manager.generate_dataset(10)
    
    # Generate visualizations
    if args.visualize:
        print("Generating visualizations...")
        visualizer.plot_poverty_map(df)
        visualizer.plot_poverty_distribution(df)
        
        # Generate report cards for top poverty locations
        print("Generating location report cards...")
        for _, row in df.sort_values('poverty_score', ascending=False).head(5).iterrows():
            # Get satellite image
            img = satellite.get_location_image(row['latitude'], row['longitude'])
            
            # Create location card
            visualizer.create_location_card(
                row['city'],
                row['country'],
                row['poverty_score'],
                img
            )
    
    # Train model if requested
    if args.train:
        print("Training CNN model...")
        # Extract images and poverty scores
        images = []
        scores = []
        
        print("Preparing training data...")
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"Processing image {i+1}/{len(df)}", end='\r')
            img = satellite.get_location_image(row['latitude'], row['longitude'])
            images.append(img)
            scores.append(row['poverty_score'])
        print("\nData preparation complete.")
        
        # Train the model
        result = model_manager.train(images, scores, epochs=args.epochs)
        print(f"Model training complete. Training MAE: {result['training_accuracy']:.4f}, Validation MAE: {result['validation_accuracy']:.4f}")
    
    # Analyze specific location if requested
    if args.analyze:
        try:
            lat, lon = map(float, args.analyze.split(','))
            print(f"Analyzing location at coordinates: {lat}, {lon}")
            
            # Get location info
            location = data_manager._get_location_info(lat, lon)
            
            # Get satellite image
            img = satellite.get_location_image(lat, lon)
            
            # Generate prediction and explanation
            explanation = model_manager.explain_prediction(img)
            
            # Display results
            print(f"Location: {location['city']}, {location['country']}")
            print(f"Predicted poverty score: {explanation['poverty_score']:.1f}/100")
            print(f"Poverty level: {data_manager._classify_poverty(explanation['poverty_score'])}")
            print(f"Visual explanation saved to: {explanation['visual_explanation_path']}")
            
            # Create location card
            visualizer.create_location_card(
                location['city'],
                location['country'],
                explanation['poverty_score'],
                img
            )
            
            # Print feature importance
            print("\nFeature Importance:")
            for feature, importance in explanation['feature_importance'].items():
                print(f"  {feature}: {importance:.2f}")
        
        except Exception as e:
            print(f"Error analyzing location: {e}")
    
    # If no specific action is requested, show help
    if not (args.generate or args.visualize or args.train or args.analyze):
        parser.print_help()

if __name__ == "__main__":
    main()
















