import sqlite3
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s')

def setup_database():
    try:
        # Load Excel data
        df = pd.read_excel('./data/dog_images_data_table.xls')
        logging.info(f"Loaded Excel data: {len(df)} images")
        
        # Keep only required columns
        if not all(col in df.columns for col in ['Image_File_Path', 'Breed']):
            raise KeyError("Required columns 'Image_File_Path' and 'Breed' not found in Excel file")
        df = df[['Image_File_Path', 'Breed']].copy()
        
        # Normalize Image_File_Path
        df['Image_File_Path'] = df['Image_File_Path'].apply(lambda x: os.path.relpath(x, start='.'))
        
        # Map breeds to labels
        breed_to_label = {
            'Dalmatian': 0,
            'Golden Retriever': 1,
            'Pug': 2,
            'Rottweiler': 3
        }
        df['Label'] = df['Breed'].map(breed_to_label)
        if df['Label'].isnull().any():
            raise ValueError("Some breeds could not be mapped to labels. Check 'Breed' column values.")
        
        # Assign splits (70% train, 15% val, 15% test)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Breed'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Breed'])
        train_df['Split'] = 'train'
        val_df['Split'] = 'val'
        test_df['Split'] = 'test'
        
        # Combine splits
        df_split = pd.concat([train_df, val_df, test_df])
        
        # Save to dataset_splits.csv
        df_split.to_csv('./data/dataset_splits.csv', index=False)
        logging.info("Saved dataset_splits.csv")
        
        # Connect to SQLite database
        conn = sqlite3.connect('./data/DogImages.db')
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                Image_File_Path TEXT PRIMARY KEY,
                Breed TEXT,
                Label INTEGER,
                Split TEXT
            )
        ''')
        
        # Insert data
        df_split[['Image_File_Path', 'Breed', 'Label', 'Split']].to_sql('images', conn, if_exists='replace', index=False)
        logging.info("Images table created and populated")
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                Image_File_Path TEXT PRIMARY KEY,
                Predicted_Label TEXT,
                Confidence REAL
            )
        ''')
        logging.info("Predictions table created")
        
        conn.commit()
        conn.close()
        logging.info("Database setup complete: ./data/DogImages.db")
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        raise e

if __name__ == "__main__":
    setup_database()