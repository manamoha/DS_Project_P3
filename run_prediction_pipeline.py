from load_data import load_data
from preprocess import create_data_generator
from feature_engineering import apply_feature_engineering
from make_predictions import make_predictions
import os
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_prediction_pipeline():
    new_images_dir = './new_images'
    
    try:
        logging.info("Starting prediction pipeline")
        
        # Data Loading
        df = load_data('predict', new_images_dir=new_images_dir)
        
        # Data Preprocessing
        generator = create_data_generator(df, 'predict')
        
        # Feature Engineering
        generator = apply_feature_engineering(df, generator, 'predict')
        
        # Prediction
        results = make_predictions(generator, df['Image_File_Path'].tolist())
        
        print(f"\nPrediction Pipeline Completed")
        print(f"Predictions saved to ./predictions.csv and database")
        logging.info("Prediction pipeline completed successfully")
    except Exception as e:
        logging.error(f"Prediction pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    run_prediction_pipeline()