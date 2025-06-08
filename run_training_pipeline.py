from load_data import load_data
from preprocess import create_data_generator
from feature_engineering import apply_feature_engineering
from train_model import train_model
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s')

def run_training_pipeline():
    try:
        logging.info("Starting training pipeline")
        
        # Data Loading
        train_df = load_data('train')
        val_df = load_data('val')
        test_df = load_data('test')

        
        # Data Preprocessing
        # train_generator = create_data_generator(train_df, 'train')
        # val_generator = create_data_generator(val_df, 'val')
        # test_generator = create_data_generator(test_df, 'test')
        
        # Feature Engineering
        train_generator = apply_feature_engineering(train_df, None, 'train')
        val_generator = apply_feature_engineering(val_df, None, 'val')
        test_generator = apply_feature_engineering(test_df, None, 'test')
        
        # Modeling
        model, history, val_acc, val_f1, test_acc, test_f1 = train_model(
            train_generator, val_generator, train_df, val_df, test_generator, test_df
        )
        
        print(f"\nTraining Pipeline Completed")
        print(f"VGG16 Validation Accuracy: {val_acc:.4f}, F1-Score: {val_f1:.4f}")
        if test_acc is not None:
            print(f"VGG16 Test Accuracy: {test_acc:.4f}, F1-Score: {test_f1:.4f}")
        logging.info("Training pipeline completed successfully")
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    run_training_pipeline()