import tensorflow as tf
import pandas as pd
import numpy as np
import sqlite3
import logging
import mlflow.tensorflow

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_predictions(generator, image_paths, run_id=None):
    class_labels = ['Dalmatian', 'Golden Retriever', 'Pug', 'Rottweiler']
    
    try:
        # Load model
        if run_id:
            model_uri = f"runs:/{run_id}/vgg16_model"
            model = mlflow.tensorflow.load_model(model_uri)
            logging.info(f"Loaded VGG16 model from MLFlow run_id: {run_id}")
        else:
            model = tf.keras.models.load_model('./vgg16_model.h5')
            logging.info("Loaded VGG16 model from local file")
        
        # Predict
        predictions = model.predict(generator, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        pred_labels = [class_labels[idx] for idx in pred_classes]
        pred_probs = np.max(predictions, axis=1)
        
        # Save to DataFrame
        results = pd.DataFrame({
            'Image_File_Path': image_paths,
            'Predicted_Label': pred_labels,
            'Confidence': pred_probs
        })
        results.to_csv('./predictions.csv', index=False)
        logging.info("Predictions saved to ./predictions.csv")
        
        # Save to database
        conn = sqlite3.connect('./data/DogImages.db')
        results.to_sql('predictions', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        logging.info("Predictions saved to database")
        
        return results
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise e