import sqlite3
import pandas as pd
import os
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_type='train', new_images_dir=None):
    try:
        conn = sqlite3.connect('./data/DogImages.db')
        if data_type in ['train', 'val', 'test']:
            query = f"SELECT Image_File_Path, Breed, Label FROM images WHERE Split = '{data_type}'"
            df = pd.read_sql_query(query, conn)
            df['Label_str'] = df['Label'].astype(str)
            logging.info(f"Loaded {data_type} data: {len(df)} images")
        elif data_type == 'predict' and new_images_dir:
            image_files = [f for f in os.listdir(new_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            df = pd.DataFrame({
                'Image_File_Path': [os.path.join(new_images_dir, f) for f in image_files],
                'Breed': [None] * len(image_files),
                'Label': [None] * len(image_files),
                'Label_str': ['0'] * len(image_files)  # Dummy labels
            })
            df['Image_File_Path'] = df['Image_File_Path'].apply(lambda x: os.path.relpath(x, start='.'))
            logging.info(f"Loaded new images for prediction: {len(df)} images")
        else:
            raise ValueError("Invalid data_type or missing new_images_dir")
        
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        raise e