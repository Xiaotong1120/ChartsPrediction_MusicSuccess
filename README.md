# Spotify Data Analysis Project

## Overview
This repository contains a comprehensive data analysis project that processes and analyzes Spotify chart data using Apache Spark and machine learning techniques. The project identifies what audio features and characteristics contribute to a song's chart success, implementing a complete pipeline from initial data cleaning through advanced predictive modeling and business recommendations. The system ultimately generates actionable insights for music marketing teams to prioritize song promotion based on chart potential.

## Files
1. **DataManipulation.ipynb**: Initial data cleaning, transformation, and deduplication
2. **EDA.ipynb**: Exploratory data analysis and classification setup
3. **InitialModel.ipynb**: Feature engineering, model training, and evaluation
4. **FinalModel.ipynb**: Advanced feature engineering, ensemble modeling, and deployment

## DataManipulation.ipynb
This notebook performs the initial data processing steps:

### Features
- **Data Loading**: 
  - Loads Spotify chart data from Google Cloud Storage
  - Implements a properly defined schema with 29 strongly-typed columns
  - Handles large dataset efficiently with Spark optimizations
- **Data Exploration**: 
  - Examines the structure and size of the dataset (26.2 million rows initially)
  - Analyzes column types and distributions
  - Identifies 200,290 unique tracks across global charts
- **Data Cleaning**: 
  - Removes 9 unnecessary columns that don't contribute to analysis
  - Deduplicates tracks based on track_id to eliminate redundancy
  - Preserves critical attributes including audio features
- **Data Export**: 
  - Saves the cleaned and deduplicated dataset to Google Cloud Storage
  - Prepares data in optimized format for subsequent analysis
  - Maintains data integrity throughout transformation process

### Technical Details
- Uses **PySpark** for efficient distributed processing of large-scale data
- Implements memory optimization techniques:
  - Off-heap memory configuration (2GB)
  - Explicit schema definition to avoid type inference overhead
  - Caching of intermediate results for performance
- Reduces dataset size by ~99% through deduplication while preserving unique tracks
  - From 26,174,269 rows to 200,290 unique tracks
  - Preserves all 12 audio features for each track
- Processes over 26 million rows with optimized Spark DataFrame operations
- Implements robust error handling for data type conversions

### Data Transformation Results
- **Initial Data**: 26,174,269 rows with 29 columns
- **Cleaned Data**: 26,174,269 rows with 20 columns (removed 9 unnecessary columns)
- **Final Deduplicated Data**: 200,290 unique tracks with 20 columns
- **Deduplication Rate**: 99.23% reduction in data volume
- **Data Schema**: Structured format with proper typing for numerical features (12 audio features)
- **Output Format**: Efficiently stored CSV file in Google Cloud Storage

## EDA.ipynb
This notebook performs exploratory data analysis and classification setup to identify factors that influence song popularity:

### Features
- **Data Integration**: 
  - Combines deduplicated rank data with comprehensive track features dataset
  - Merges track metadata with audio characteristics
  - Links chart performance data with song features
- **Artist Filtering**: 
  - Extracts 75,315 unique artists who have appeared in rankings
  - Implements substring matching to identify artists across datasets
  - Filters dataset to focus on tracks from charting artists
- **Classification Setup**: 
  - Creates a binary classification problem (ranked vs. non-ranked songs)
  - Develops feature sets for machine learning
  - Prepares training and testing datasets
- **Data Processing**: 
  - Handles data from 2015-2021, focusing on relevant time periods
  - Filters to relevant time periods for chart consistency
  - Extracts and transforms features for analysis

### Technical Details
- Uses **Pandas** and **NumPy** for data manipulation
- Implements **Scikit-learn** for machine learning models:
  - Logistic Regression for baseline performance
  - Random Forest for complex feature interactions
- Includes evaluation metrics:
  - ROC-AUC to measure discrimination ability
  - Classification reports with precision/recall metrics
  - Confusion matrices to understand error patterns
- Generates visualization output to understand data patterns

### Dataset Composition
- **Rank dataset**: 197,940 rows of chart performance data
  - Contains chart positions, dates, regions, and trends
  - Includes unique track identifiers and metadata
- **Track features dataset**: 1,204,025 songs (filtered to 338,462 from 2015-2021)
  - Contains audio features from Spotify's API
  - Includes release information and track metadata
- **Final analysis dataset**: 310,760 songs by ranked artists
  - **7,526 songs that appeared in rankings** (2.42%)
  - **303,234 songs that did not appear in rankings** (97.58%)
  - Extreme class imbalance presenting modeling challenges

## InitialModel.ipynb
This notebook performs comprehensive exploratory data analysis, feature engineering, model training and evaluation:

### Features
- **Exploratory Data Analysis**: 
  - In-depth examination of audio features and their distributions
  - Statistical analysis of feature ranges and outliers
  - Investigation of top artists and popular tracks
- **Correlation Analysis**: 
  - Creates correlation matrices between all audio features
  - Identifies key relationships (e.g., energy-loudness correlation of 0.74)
  - Visualizes feature relationships using heatmaps
- **Data Visualization**: 
  - Creates histograms of all audio features to understand distributions
  - Generates boxplots to compare feature values across explicit/non-explicit content
  - Implements PCA visualization to understand feature spaces
- **Feature Engineering**: 
  - Prepares and standardizes features for machine learning models
  - Creates feature vectors for model training
  - Handles missing values and outliers
- **Model Training**: 
  - Implements multiple classification algorithms:
    - Logistic Regression
    - Random Forest
    - Gradient Boosted Trees
    - Decision Trees
    - Linear SVM
    - Naive Bayes
    - Neural Networks (Multilayer Perceptron)
  - Creates ensemble voting model combining individual predictions
- **Model Evaluation**: 
  - Evaluates model performance using comprehensive metrics:
    - Area Under ROC Curve (AUC)
    - F1 Score
    - Precision and Recall
    - Confusion Matrices
  - Analyzes feature importance from tree-based models
  - Examines model errors to understand misclassifications

### Technical Details
- Uses **PySpark** for efficient processing of large-scale data
- Implements data caching for performance optimization
- Uses StandardScaler for feature normalization
- Handles class imbalance through sampling approaches
- Evaluates models with comprehensive metrics (accuracy, precision, recall, F1)
- Carefully handles class imbalance (only 2.69% of songs appear in rankings)

### Results
- Achieves AUC scores up to 0.814 for predicting track chart performance
- Random Forest classifier performs best among the tested models
- Identifies key audio features that distinguish ranked vs. non-ranked tracks
- Analyzes confusion matrices to understand model errors
- Discovers model limitations with highly imbalanced data

## Dataset Schema
The dataset contains rich information about tracks and their audio characteristics:

### Basic Track Information
- **id**: Unique identifier for the track in Spotify's catalog
- **title/name**: Song title
- **artist**: Artist name(s), may include multiple artists separated by commas
- **album**: Album name the track appears on
- **track_id**: Spotify's unique track identifier used for deduplication
- **release_date**: When the track was released
- **year**: Extracted year from release date (used for filtering)

### Chart Metrics
- **rank**: Position on the chart (1-200)
- **date**: Date the track appeared on the chart
- **region**: Country or region where the chart was compiled
- **chart**: Chart type (e.g., "top200", "viral50")
- **trend**: Trend indicator (e.g., "MOVE_UP", "MOVE_DOWN", "SAME_POSITION")
- **streams**: Number of streams for the track in that region and date
- **popularity**: Spotify's internal popularity metric (0-100)

### Track Metadata
- **duration_ms**: Duration of the track in milliseconds
- **explicit**: Boolean flag indicating explicit content
- **available_markets**: List of markets where the track is available

### Audio Features (from Spotify API)
- **danceability**: How suitable the track is for dancing (0.0-1.0)
- **energy**: Perceptual measure of intensity and activity (0.0-1.0)
- **key**: Estimated key of the track (0-11 integers matching pitch class notation)
- **loudness**: Overall loudness in decibels (dB) (typically -60 to 0)
- **mode**: Modality of the track (0 = minor, 1 = major)
- **speechiness**: Presence of spoken words (0.0-1.0)
- **acousticness**: Confidence measure of acoustic quality (0.0-1.0)
- **instrumentalness**: Predicts whether a track has no vocals (0.0-1.0)
- **liveness**: Detects presence of audience (0.0-1.0)
- **valence**: Musical positiveness conveyed by the track (0.0-1.0)
- **tempo**: Estimated tempo in beats per minute (BPM)
- **time_signature**: Estimated time signature (3-7 indicating beats per measure)

### Derived Features (Created in FinalModel.ipynb)
The project creates 18 additional features derived from the base audio features:

#### Music Theory Features
- **energy_acoustic_ratio**: Ratio of electronic vs. acoustic sound
- **dance_valence_product**: Distinguishes happy vs. sad dance music
- **vocal_instrumental_balance**: Ratio of vocal to instrumental content

#### Audience Perception Features
- **rhythm_factor**: Combination of tempo and danceability
- **mood_intensity**: Combination of valence and energy
- **calmness_factor**: Composite measure of acoustic relaxed qualities

#### Categorical Features
- **is_instrumental**: Binary flag for strong instrumental tracks
- **is_rhythmic**: Binary flag for highly danceable tracks
- **is_energetic**: Binary flag for high-energy tracks
- **is_happy**: Binary flag for high-valence tracks

#### Non-linear Transformations
- **log_acousticness**: Logarithmic transformation of acousticness
- **log_instrumentalness**: Logarithmic transformation of instrumentalness
- **loudness_squared**: Loudness raised to power of 2
- **loudness_cubic**: Loudness raised to power of 3

#### Composite Indicators
- **mainstream_index**: Combined popularity indicators
- **experimental_index**: Non-mainstream feature combinations
- **loudness_energy_interaction**: Interaction term of loudness and energy
- **vocal_clarity**: Measure of vocal presence and clarity

## Requirements
### Core Technologies
- **Apache Spark** 3.x (for distributed data processing)
- **Python** 3.x (for scripting and analysis)
- **PySpark SQL** (for DataFrame operations)
- **PySpark ML** (for machine learning model building)
- **Pandas** (for local data analysis)
- **NumPy** (for numerical operations)
- **Scikit-learn** (for machine learning algorithms)

### Data Visualization
- **Matplotlib** (for static visualizations)
- **Seaborn** (for statistical visualizations)

### Cloud Resources
- **Google Cloud Storage (GCS)** (for data storage)
- **Google Cloud Dataproc** (for distributed processing)
  - Cluster with at least 3 nodes recommended
  - Worker node configuration: 16GB RAM, 4 vCPUs

### System Requirements
- At least 8GB RAM for local development
- 50GB+ disk space for data storage
- Support for off-heap memory allocation

### Python Libraries
```
pyspark==3.3.0
pandas==1.5.0
numpy==1.23.0
matplotlib==3.6.0
seaborn==0.12.0
scikit-learn==1.1.0
google-cloud-storage==2.5.0
```

## Usage
### Setting Up the Environment

1. **Configure Google Cloud Platform**:
   ```bash
   # Install Google Cloud SDK
   curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-367.0.0-linux-x86_64.tar.gz
   tar -xzf google-cloud-sdk-367.0.0-linux-x86_64.tar.gz
   ./google-cloud-sdk/install.sh
   
   # Authenticate
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. **Create a Dataproc Cluster**:
   ```bash
   gcloud dataproc clusters create spotify-analysis-cluster \
     --region=us-central1 \
     --master-machine-type=n1-standard-4 \
     --worker-machine-type=n1-standard-4 \
     --num-workers=3 \
     --image-version=2.0-debian10
   ```

3. **Start Jupyter on the Cluster**:
   ```bash
   gcloud dataproc jobs submit pyspark --cluster=spotify-analysis-cluster \
     --properties=spark.jars.packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 \
     --driver-log-levels=root=INFO \
     --region=us-central1 \
     --files=gs://your-bucket/notebooks/requirements.txt \
     -- -m jupyter notebook
   ```

### Running the Notebooks

Run the notebooks in sequential order to build the complete analysis pipeline:

1. **Data Preparation** (`DataManipulation.ipynb`):
   - Upload your Spotify chart data to GCS
   - Update file paths in the notebook to point to your data
   - Execute all cells to clean and deduplicate the data
   - Check the output path for the deduplicated dataset

2. **Exploratory Analysis** (`EDA.ipynb`):
   - Update the input path to point to your deduplicated data
   - Run all cells to perform initial data analysis
   - Review the visualizations and summary statistics

3. **Model Development** (`InitialModel.ipynb`):
   - Update data paths to point to your prepared data
   - Run all cells to build and evaluate initial models
   - Note the best performing model configuration

4. **Advanced Modeling** (`FinalModel.ipynb`):
   - Update all file paths to point to your data
   - Run the feature engineering cells
   - Execute the balanced sampling code
   - Run model training cells for all model types
   - Execute the tiered prediction system cells
   - Run the marketing recommendation code
   - Check the output directories for saved models and recommendations

### Using the Production System

For deploying the song prediction system in production:

1. **Load Saved Models**:
   ```python
   from pyspark.ml.classification import RandomForestClassificationModel
   
   # Load models
   rf_high_recall_model = RandomForestClassificationModel.load("gs://your-bucket/spotify_rf_high_recall_final_model")
   rf_balanced_model = RandomForestClassificationModel.load("gs://your-bucket/spotify_rf_balanced_final_model")
   rf_high_precision_model = RandomForestClassificationModel.load("gs://your-bucket/spotify_rf_high_precision_final_model")
   gbt_model = GBTClassificationModel.load("gs://your-bucket/spotify_gbt_final_model")
   ```

2. **Process New Songs**:
   ```python
   # Load feature engineering functions
   # Apply same feature transformations
   # Use the tiered_predictions function
   predictions = create_tiered_predictions(
       new_songs_df, 
       rf_high_recall_model, 
       rf_balanced_model, 
       rf_high_precision_model, 
       gbt_model, 
       0.30  # Best threshold from training
   )
   ```

3. **Apply Marketing Recommendations**:
   ```python
   # Filter by confidence level
   tier_a_songs = predictions.filter(col("confidence_level").contains("Tier A"))
   
   # Export for marketing team
   tier_a_songs.select("name", "artists", "hit_score", "marketing_recommendation").write.csv("priority_songs.csv")
   ```

## FinalModel.ipynb
This notebook represents the culmination of the project with advanced modeling techniques and business-focused applications:

### Features
- **Enhanced Feature Engineering**: Creates 18 sophisticated derived features:
  - **Music Theory Features**: Energy-to-acoustic ratio, dance-valence product, vocal-instrumental balance
  - **Audience Perception Features**: Rhythm factor, mood intensity, calmness factor
  - **Categorical Flags**: Binary indicators for instrumental, rhythmic, energetic, and happy tracks
  - **Non-linear Transformations**: Log-transformed acousticness and instrumentalness, loudness squared and cubed
  - **Composite Indicators**: Mainstream popularity index, experimental index, loudness-energy interaction, vocal clarity
- **Balanced Sampling Strategies**: Implements three different ratios to address extreme class imbalance (97.31% non-ranked songs):
  - 1:2 balanced datasets (high recall optimization)
  - 1:3 balanced datasets (balanced precision-recall)
  - 1:5 balanced datasets (high precision optimization)
- **Multi-Model Ensemble System**: Develops a tiered prediction system combining strengths of multiple models:
  - Random Forest models with different balancing ratios
  - Gradient Boosting Trees with optimized thresholds
  - Weighted voting ensemble for final predictions
- **Threshold Optimization**: Fine-tunes decision thresholds across probability ranges (0.05-0.30) to maximize F2 score
- **Business-Oriented Output**: Generates structured marketing recommendations with confidence tiers and ROI estimates

### Technical Details
- Uses **PySpark ML** for highly scalable machine learning on the full dataset
- Implements feature importance analysis to identify key predictors of chart success
- Creates specialized evaluation metrics focused on business value (F2 score prioritizing recall)
- Develops a confidence-based prediction system with four distinct tiers
- Generates comprehensive marketing recommendations with business justifications
- Exports results to CSV format for marketing team consumption

### Implementation Highlights
- **Dynamic Thresholding**: Programmatically tests multiple probability thresholds to find optimal decision boundaries
- **Weighted Ensemble Voting**: Combines predictions using a weighted scheme (0.4, 0.3, 0.2, 0.1) for model integration
- **Confidence-Level Assignment**: Uses weighted probabilities to assign songs to appropriate marketing tiers
- **Comprehensive Evaluation**: Analyzes model performance across multiple metrics (accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC)
- **Error Analysis**: Examines false positives and false negatives to understand model limitations

### Business Results
- Achieves high model performance with F1 scores up to 0.188 and AUC scores of 0.824
- Creates a four-tier classification system with clear performance differentiation:
  - **Tier A (Highly Likely to Chart)**: 26.58% actual chart rate - recommended for full promotion
  - **Tier B (Moderately Likely to Chart)**: 10.93% actual chart rate - moderate promotion
  - **Tier C (Low Likelihood to Chart)**: 4.70% actual chart rate - limited promotion
  - **Tier D (Unlikely to Chart)**: 0.81% actual chart rate - basic support only
- Identifies the most important features for chart prediction:
  - loudness_squared (0.077)
  - experimental_index (0.076)
  - vocal_instrumental_balance (0.075)
  - instrumentalness (0.073)
  - danceability (0.070)
- Provides detailed marketing recommendations with justifications and ROI estimates
- Exports predictions as a structured CSV ready for business implementation

## Acknowledgments
- **Data Sources**:
  - Spotify Charts API for global chart data
  - Spotify Web API for detailed audio features
  - Spotify for Developers platform for documentation and support

- **Technologies**:
  - Apache Spark community for the powerful distributed computing framework
  - Google Cloud Platform for scalable infrastructure
  - PySpark ML library for distributed machine learning capabilities
  - Scikit-learn for foundational machine learning algorithms

- **Research References**:
  - Schedl, M., & Hauger, D. (2015). "Tailoring music recommendations to users by considering diversity, mainstreamness, and novelty."
  - Interiano, M., et al. (2018). "Musical trends and predictability of success in contemporary songs in and out of the top charts."
  - Dhanaraj, R., & Logan, B. (2005). "Automatic prediction of hit songs."

This project builds upon the growing field of music information retrieval (MIR) and computational approaches to understanding music popularity. We acknowledge the researchers who have established foundational methods in this domain.
