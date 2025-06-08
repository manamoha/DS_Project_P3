from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_feature_engineering(df, generator, data_type='train'):
    try:
        if data_type == 'train':
            aug_datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True,
                fill_mode='nearest',
                rescale=1./255
            )
            aug_generator = aug_datagen.flow_from_dataframe(
                dataframe=df,
                directory=None,
                x_col='Image_File_Path',
                y_col='Label_str',
                target_size=(224, 224),
                batch_size=16,
                class_mode='sparse',
                shuffle=True,
                validate_filenames=True
            )
            logging.info("Applied augmentation for training data")
            return aug_generator
        else:
            val_test_datagen = ImageDataGenerator(rescale=1./255)
            val_test_generator = val_test_datagen.flow_from_dataframe(
                dataframe=df,
                directory=None,
                x_col='Image_File_Path',
                y_col='Label_str',
                target_size=(224, 224),
                batch_size=16,
                class_mode='sparse',
                shuffle=False,
                validate_filenames=True
            )
            logging.info("No augmentation applied for non-training data")
            return val_test_generator
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        raise e