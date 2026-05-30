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

The project uses `Spotify_Dataset_V3.csv` containing **1,000+ Spotify chart records** with fields including:
- **Song Information**: Title, Artists, Song URL
- **Audio Features** (7 dimensions): Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Valence
- **Metadata**: Rank, Date, Nationality, Continent, Points
- **Derived Features**: Song count, total points, days on chart

## 🔍 Key Features

### Data Processing
- CSV delimiter conversion for compatibility
- Data normalization using Z-score method
- Feature extraction and aggregation
- Duplicate removal and data cleaning
- ~95% data retention after cleaning

### Visualization
- **Yearly Trends**: Track how audio features evolve over 15+ years
- **Geographic Analysis**: Visualize song distribution across 50+ countries and 6 continents
- **Artist Analysis**: Monitor top 100+ artists' presence in charts over time
- **Ranking Analysis**: Examine feature correlations with chart rankings

### Machine Learning Models

The project implements **5 machine learning models** for binary classification (e.g., predicting if a song will have above/below median song count):

- **Random Forest (RDF)**: Ensemble-based classification (100-200 trees)
- **Support Vector Machine (SVM)**: Kernel-based classification with RBF kernel
- **Multi-Layer Perceptron (MLP)**: Neural network classifier with 2+ hidden layers
- **TensorFlow Feed-Forward Network (TF_FFN)**: Deep learning model with optimized architecture
- **Dummy Predictor**: Baseline model for comparison (50% accuracy)

**Expected Model Performance**:
- Baseline accuracy: 50%
- Target accuracy: 70-85%
- Models trained on 7 audio features + 3 derived features

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
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
Generates 20+ plots including time-series analysis, geographic distributions, and feature correlations.

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
- Track trends for 7 audio features across 15+ years
- Monitor percentage changes in song distribution across 6 continents
- Analyze artist popularity evolution for 100+ top artists

### Feature Engineering
- **Song count**: Number of times a song appears in dataset (range: 1-50+ appearances)
- **Total points**: Aggregate ranking points per song (range: 0-10,000+)
- **Days on chart**: Duration from first appearance to reference date (range: 0-5,000+ days)

## 🤖 Model Configuration

Models are configured with parameters like:
- `label_type`: Target variable (e.g., "song_count")
- `binary`: Whether to use binary or multi-class classification
- `threshold`: Classification threshold for binary problems (default: 135 appearances)
- `test_size`: Train-test split ratio (typically 80-20 or 70-30)

## 📁 Output

- **Plots**: 20+ generated visualizations saved in `plots/` directory
- **Models**: 5 trained models stored in `best_models/` directory
- **Extracted Data**: Processed features (1,000+ records × 10 features) saved in `extracted_data/` directory

## 📈 Expected Results

- **Data Coverage**: 1,000+ Spotify chart records
- **Geographic Scope**: 50+ countries across 6 continents
- **Time Period**: 15+ years of chart history
- **Feature Dimensions**: 10 features per record
- **Model Accuracy**: 70-85% (depending on model and task)
- **Processing Time**: ~2-10 minutes depending on model complexity

## 📝 Notes

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

**Last Updated**: May 2026
