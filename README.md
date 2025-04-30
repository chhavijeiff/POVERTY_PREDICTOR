# Satellite Imagery Poverty Prediction System

A machine learning system that uses CNN models to analyze satellite imagery and predict poverty levels in different regions around the world.

## Overview

This project uses deep learning techniques to analyze satellite imagery and predict poverty levels in different regions. The system incorporates:

1. **Satellite Image Processing**: Fetches and processes satellite imagery from Mapbox or generates simulated images when API access is unavailable.

2. **CNN Model**: A custom deep learning model to extract features from satellite images and predict poverty scores.

3. **Data Management**: Tools to create, manage, and analyze datasets of locations with poverty indicators.

4. **Visualization**: Creates interactive maps, charts, and location report cards to visualize poverty data.

## Features

- Fetch high-resolution satellite imagery from Mapbox API (or generate simulated imagery)
- Train and evaluate CNN models for poverty prediction
- Generate synthetic datasets for testing and demonstration
- Create interactive poverty maps and visualizations
- Generate location-specific poverty analysis reports
- Explainable AI features to understand predictions

## Directory Structure

```
poverty-prediction-system/
│
├── modules/                  # Core modules
│   ├── __init__.py           # Module initialization
│   ├── satellite_imagery.py  # Satellite image fetching & simulation
│   ├── model_manager.py      # CNN model management
│   ├── data_manager.py       # Dataset management
│   └── visualization.py      # Visualization tools
│
├── data/                     # Data storage
│   └── poverty_dataset.csv   # Location dataset
│
├── image_cache/              # Cached satellite images
│
├── models/                   # Trained models
│   └── poverty_cnn_model.h5  # Saved CNN model
│
├── output/                   # Generated output
│   ├── charts/               # Generated charts and visualizations
│   ├── location_cards/       # Location report cards
│   └── model_results/        # Model training results
│
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/poverty-prediction-system.git
cd poverty-prediction-system
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Set your Mapbox API token:
```bash
export MAPBOX_TOKEN=your_mapbox_token_here
```

## Usage

### Generate a dataset:
```bash
python main.py --generate 50
```

### Visualize the dataset:
```bash
python main.py --visualize
```

### Train the CNN model:
```bash
python main.py --train --epochs 10
```

### Analyze a specific location:
```bash
python main.py --analyze "37.7749,-122.4194"
```

### Combine operations:
```bash
python main.py --generate 20 --visualize --train
```

## CNN Model Architecture

The system uses a custom CNN architecture with:
- 3 convolutional blocks with increasing filter sizes (32→64→128)
- Batch normalization for training stability
- Max pooling layers for spatial reduction
- Dropout layers to prevent overfitting
- Dense layers for final poverty score prediction

## Simulated Satellite Imagery

When Mapbox API access is unavailable, the system can generate realistic satellite imagery simulations with:
- Urban/rural differentiation
- Building density variations
- Road networks
- Vegetation and water features
- Poverty-correlated visual elements

## Explanation Features

The model provides explanations for its predictions through:
- Visual attention maps highlighting areas that influenced the prediction
- Feature importance scores for different visual elements
- Location report cards with detailed analysis

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- NumPy, Pandas, Matplotlib, Seaborn
- Pillow, Requests
- Scikit-learn
- GeoPy
- Plotly

## License

This project is licensed under the MIT License - see the LICENSE file for details.