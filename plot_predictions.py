import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_predictions():
    try:
        logging.info("Starting prediction visualization")

        # Load predictions
        predictions_df = pd.read_csv('./predictions.csv')
        predictions_df['Image_File_Path'] = predictions_df['Image_File_Path'].apply(os.path.normpath)

        # Load real breeds
        real_breeds_df = pd.read_excel('./Prediction_Real_Breeds.xlsx')
        real_breeds_df['Image_File_Path'] = real_breeds_df['Image_File_Path'].apply(os.path.normpath)

        # Selected images (corrected filenames)
        selected_images = [
            'new_images/1.jpg', 'new_images/2.jpg', 'new_images/4.jpeg',
            'new_images/5.jpg', 'new_images/7.jpg', 'new_images/9.jpg',
            'new_images/13.jpg', 'new_images/14.jpg', 'new_images/15.png',
            'new_images/17.jpg', 'new_images/18.jpg', 'new_images/19.jpg'
        ]
        selected_images = [os.path.normpath(img) for img in selected_images]

        # Merge data
        merged_df = predictions_df.merge(real_breeds_df, on='Image_File_Path', how='inner')
        selected_df = merged_df[merged_df['Image_File_Path'].isin(selected_images)]
        
        # Ensure order matches selected_images
        selected_df['Image_File_Path'] = pd.Categorical(
            selected_df['Image_File_Path'], categories=selected_images, ordered=True
        )
        selected_df = selected_df.sort_values('Image_File_Path')
        
        if len(selected_df) != 12:
            missing = set(selected_images) - set(selected_df['Image_File_Path'])
            logging.error(f"Missing images in predictions: {missing}")
            raise ValueError(f"Expected 12 images, found {len(selected_df)}")

        # Create 4x3 plot
        fig, axes = plt.subplots(4, 3, figsize=(9, 12))
        axes = axes.flatten()

        for idx, row in enumerate(selected_df.itertuples()):
            # Load image
            img_path = row.Image_File_Path
            if not os.path.exists(img_path):
                logging.error(f"Image not found: {img_path}")
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path)
            
            # Plot image
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # Add title
            predicted = row.Predicted_Label
            real = row.Breed
            title = f"Predicted: {predicted}\nReal: {real}"
            axes[idx].set_title(title, fontsize=10, pad=5)

        # Adjust layout
        plt.tight_layout()
        plt.savefig('prediction_comparison.png', dpi=300)
        plt.close()
        logging.info("Prediction visualization saved: prediction_comparison.png")
        
        print("Prediction visualization completed. Saved as prediction_comparison.png")
    except Exception as e:
        logging.error(f"Prediction visualization failed: {e}")
        raise e

if __name__ == "__main__":
    plot_predictions()