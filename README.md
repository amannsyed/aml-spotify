# AML Spotify Analysis

A machine learning project for analyzing Spotify song data and building predictive models based on audio features and metadata.

## 📋 Overview

This repository contains code for analyzing Spotify's top charting songs dataset with a focus on:
- **Data Exploration & Visualization**: Analyzing trends in audio features (danceability, energy, valence, etc.) over time
- **Data Processing**: Cleaning, normalizing, and transforming Spotify data
- **Machine Learning**: Building and evaluating predictive models to classify songs based on their characteristics

## 🏗️ Project Structure

```
aml-spotify/
├── runner.py                 # Main entry point for running ML models
├── visualization.py          # Data exploration and visualization scripts
├── modules/                  # Core functionality modules
│   ├── convert_csv          # CSV data conversion utilities
│   ├── dataClass            # Data structures and classes
│   ├── plot_and_extract_data # Plotting and feature extraction
│   └── ml_models            # Machine learning model implementations
├── datasets/                # Input Spotify data
├── extracted_data/          # Processed and extracted features
├── plots/                   # Generated visualization outputs
└── best_models/             # Saved trained models
```

## 🎵 Dataset

The project uses `Spotify_Dataset_V3.csv` containing Spotify chart information with fields including:
- **Song Information**: Title, Artists, Song URL
- **Audio Features**: Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Valence
- **Metadata**: Rank, Date, Nationality, Continent, Points
- **Derived Features**: Song count, total points, days on chart

## 🔍 Key Features

### Data Processing
- CSV delimiter conversion for compatibility
- Data normalization using Z-score method
- Feature extraction and aggregation
- Duplicate removal and data cleaning

### Visualization
- **Yearly Trends**: Track how audio features evolve over years
- **Geographic Analysis**: Visualize song distribution by continent and nationality
- **Artist Analysis**: Monitor top artists' presence in charts over time
- **Ranking Analysis**: Examine feature correlations with chart rankings

### Machine Learning Models

The project implements multiple ML models for binary classification (e.g., predicting if a song will have above/below median song count):

- **Random Forest (RDF)**: Ensemble-based classification
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Multi-Layer Perceptron (MLP)**: Neural network classifier
- **TensorFlow Feed-Forward Network (TF_FFN)**: Deep learning model
- **Dummy Predictor**: Baseline model for comparison

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Required packages: pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn

### Installation

```bash
git clone https://github.com/amannsyed/aml-spotify.git
cd aml-spotify
pip install -r requirements.txt  # if available
```

### Usage

#### Run Visualizations
```python
python visualization.py
```

#### Run Machine Learning Models
Edit `runner.py` to select which model to train and uncomment the desired line:

```python
# Example: Train Random Forest
ml_models.find_best_RDF(label_type="song_count", binary=True, threshold=135)

# Or train SVM
ml_models.find_best_SVM(label_type="song_count", binary=True, threshold=135)

# Or use TensorFlow FFN
ml_models.find_best_TF_FFN(label_type="song_count", binary=True, threshold=135)
```

Then execute:
```bash
python runner.py
```

## 📊 Analysis Capabilities

### Feature Normalization
The project applies Z-score normalization to audio features for standardization:
```python
norm_base[column] = (norm_base[column] - norm_base[column].mean()) / norm_base[column].std()
```

### Time-Series Analysis
- Track trends for individual audio features across years
- Monitor percentage changes in song distribution by geography
- Analyze artist popularity evolution

### Feature Engineering
- Song count: Number of times a song appears in dataset
- Total points: Aggregate ranking points per song
- Days on chart: Duration from first appearance to reference date

## 🤖 Model Configuration

Models are configured with parameters like:
- `label_type`: Target variable (e.g., "song_count")
- `binary`: Whether to use binary or multi-class classification
- `threshold`: Classification threshold for binary problems

## 📁 Output

- **Plots**: Generated visualizations saved in `plots/` directory
- **Models**: Trained models stored in `best_models/` directory
- **Extracted Data**: Processed features saved in `extracted_data/` directory

## 📝 Notes

- The repository has merge conflicts in `runner.py` that should be resolved
- Uncomment sections in `runner.py` and `visualization.py` based on desired analysis
- Ensure `Spotify_Dataset_V3.csv` is placed in the `datasets/` directory

## 👤 Author

**amannsyed** - [GitHub Profile](https://github.com/amannsyed)

## 📄 License

This project is not currently licensed. See repository settings for more information.

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues or questions, please create an [Issue](https://github.com/amannsyed/aml-spotify/issues) on the repository.

---

**Last Updated**: September 2024
