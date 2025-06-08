from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_data_generator(df, data_type='train', batch_size=16, target_size=(224, 224)):
    try:
        datagen = ImageDataGenerator(rescale=1./255)
        shuffle = True if data_type == 'train' else False
        
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=None,  # Paths are relative
            x_col='Image_File_Path',
            y_col='Label_str',
            target_size=target_size,  # Resizes to 224x224
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=shuffle,
            validate_filenames=True
        )
        logging.info(f"Data generator created for {data_type}: {generator.n} images")
        return generator
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise e