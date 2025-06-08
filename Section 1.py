import pandas as pd
import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    filename='training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load dataset
try:
    df = pd.read_csv('dataset_splits.csv')
    logging.info(f"Dataset loaded: {len(df)} images")
except Exception as e:
    logging.error(f"Failed to load dataset_splits.csv: {e}")
    raise FileNotFoundError(f"dataset_splits.csv not found: {e}")

# Log class distribution
class_dist = df.groupby(['Set', 'Label']).size().unstack(fill_value=0)
logging.info(f"Class distribution:\n{class_dist}")

# Convert Label to strings for generators
df['Label_str'] = df['Label'].astype(str)

# Prepare splits
train_df = df[df['Set'] == 'train']
val_df = df[df['Set'] == 'val']
test_df = df[df['Set'] == 'test']
logging.info(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['Label']), y=train_df['Label'])
class_weights_dict = dict(enumerate(class_weights))
logging.info(f"Class weights: {class_weights_dict}")

# Configure data augmentation
aug_config = {
    'rotation_range': 30,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}
try:
    with open('augmentation_config.txt', 'w') as f:
        for key, value in aug_config.items():
            f.write(f"{key}: {value}\n")
    logging.info("Augmentation config saved to augmentation_config.txt")
except Exception as e:
    logging.error(f"Failed to save augmentation_config.txt: {e}")

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for CNN models
base_dir = r"D:/university/sem 8/Data/Final Project/Project_P3_810100207_810100247_810102/"  # Adjust if needed
def create_data_generator(datagen, df_subset, batch_size=32, target_size=(224, 224)):
    return datagen.flow_from_dataframe(
        dataframe=df_subset,
        directory=base_dir,
        x_col='Image_File_Path',
        y_col='Label_str',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True,
        validate_filenames=True
    )

train_generator = create_data_generator(train_datagen, train_df, batch_size=32)
val_generator = create_data_generator(val_test_datagen, val_df, batch_size=32)
val_generator.shuffle = False
logging.info("Data generators configured")

# Feature extractor for Random Forest
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_base.layers:
    layer.trainable = False
feature_extractor = Model(inputs=vgg_base.input, outputs=Flatten()(vgg_base.output))

def extract_features(generator):
    features = []
    labels = []
    for batch_x, batch_y in generator:
        batch_features = feature_extractor.predict(batch_x, verbose=0)
        features.append(batch_features)
        labels.append(batch_y)
        if len(features) * generator.batch_size >= generator.n:
            break
    return np.vstack(features)[:generator.n], np.hstack(labels)[:generator.n]

X_train_rf, y_train_rf = extract_features(train_generator)
X_val_rf, y_val_rf = extract_features(val_generator)
logging.info(f"Random Forest features extracted: Train={X_train_rf.shape}, Val={X_val_rf.shape}")

# Model 1: Custom CNN (modified to match PyTorch VGG architecture)
def build_vgg_like_model(num_classes=4):
    """Build a custom VGG-like model matching the PyTorch VGG architecture."""
    image_input = Input(shape=(224, 224, 3))
    
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(image_input)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Adaptive Pooling to 7x7
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    
    # Classifier
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=image_input, outputs=x)
    return model

def train_custom_cnn():
    model = build_vgg_like_model(num_classes=4)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("Training Custom VGG...")
    history = {'loss': [], 'val_accuracy': []}
    best_acc = 0.0
    checkpoint = ModelCheckpoint('custom_cnn_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

    for epoch in range(15):
        # Train
        train_result = model.fit(
            train_generator,
            steps_per_epoch=len(train_df) // 32,
            epochs=1,
            verbose=1,
            class_weight=class_weights_dict
        )
        epoch_loss = train_result.history['loss'][0]
        history['loss'].append(epoch_loss)

        # Validate
        val_pred = model.predict(val_generator, steps=len(val_df) // 32 + 1)
        val_pred_labels = np.argmax(val_pred, axis=1)
        val_true = val_df['Label'].astype(int).values[:len(val_pred_labels)]
        val_acc = accuracy_score(val_true, val_pred_labels) * 100
        val_f1 = f1_score(val_true, val_pred_labels, average='weighted')
        history['val_accuracy'].append(val_acc)

        print(f'Custom VGG - Epoch {epoch+1}/15 - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%')
        logging.info(f'Custom VGG - Epoch {epoch+1}/15 - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model.save('custom_cnn_model.h5')
            logging.info(f"New best model saved: custom_cnn_model.h5 with Val Acc: {val_acc:.2f}%")

    print(f'Best Validation Accuracy for Custom VGG: {best_acc:.2f}%')
    logging.info(f"Best Validation Accuracy for Custom VGG: {best_acc:.2f}%")

    # Final validation metrics
    val_pred = model.predict(val_generator, steps=len(val_df) // 32 + 1)
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_true = val_df['Label'].astype(int).values[:len(val_pred_labels)]
    acc = accuracy_score(val_true, val_pred_labels)
    f1 = f1_score(val_true, val_pred_labels, average='weighted')
    logging.info(f"Custom VGG - Val Accuracy: {acc:.4f}, Val F1-Score: {f1:.4f}")

    return model, history, acc, f1

# Model 2: Random Forest Classifier
def train_random_forest():
    if len(X_train_rf) == 0:
        logging.warning("No valid training data for Random Forest")
        return None, None, 0.0, 0.0

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    logging.info("Training Random Forest...")
    model.fit(X_train_rf, y_train_rf)

    if len(X_val_rf) == 0:
        logging.warning("No valid validation data for Random Forest")
        acc, f1 = 0.0, 0.0
    else:
        val_pred = model.predict(X_val_rf)
        acc = accuracy_score(y_val_rf, val_pred)
        f1 = f1_score(y_val_rf, val_pred, average='weighted')
    logging.info(f"Random Forest - Val Accuracy: {acc:.4f}, Val F1-Score: {f1:.4f}")

    return model, None, acc, f1

# Model 3: VGG16
def train_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-5]:
        layer.trainable = False  # Fine-tune last 5 layers
    for layer in base_model.layers[-5:]:
        layer.trainable = True

    image_input = Input(shape=(224, 224, 3))
    x = base_model(image_input)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    output = Dense(4, activation='softmax')(x)

    model = Model(inputs=image_input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    logging.info("Training VGG16...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=len(train_df) // 16,
        validation_steps=len(val_df) // 16,
        class_weight=class_weights_dict,
        verbose=1
    )

    val_pred = model.predict(val_generator).argmax(axis=1)
    val_true = val_df['Label'].astype(int).values
    acc = accuracy_score(val_true, val_pred)
    f1 = f1_score(val_true, val_pred, average='weighted')
    logging.info(f"VGG16 - Val Accuracy: {acc:.4f}, Val F1-Score: {f1:.4f}")

    return model, history, acc, f1

# Train all models
results = {}
models = {}
histories = {}

print("Training Custom CNN...")
model, history, acc, f1 = train_custom_cnn()
models["Custom_CNN"] = model
histories["Custom_CNN"] = history
results["Custom_CNN"] = (acc, f1)

print("Training Random Forest...")
model, history, acc, f1 = train_random_forest()
if model is not None:
    models["Random_Forest"] = model
    histories["Random_Forest"] = history
    results["Random_Forest"] = (acc, f1)

print("Training VGG16...")
model, history, acc, f1 = train_vgg16()
models["VGG16"] = model
histories["VGG16"] = history
results["VGG16"] = (acc, f1)

# Print monitoring results
print("\nValidation Monitoring (Not Final Evaluation):")
for model_name, (acc, f1) in results.items():
    print(f"{model_name}: Validation Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")
    logging.info(f"{model_name} Monitoring: Validation Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

# Save models
try:
    models["Custom_CNN"].save("custom_cnn_model.h5")
    if "Random_Forest" in models:
        joblib.dump(models["Random_Forest"], "random_forest_model.joblib")
    models["VGG16"].save("vgg16_model.h5")
    logging.info("Models saved: custom_cnn_model.h5, random_forest_model.joblib, vgg16_model.h5")
    print("Models saved: custom_cnn_model.h5, random_forest_model.joblib, vgg16_model.h5")
except Exception as e:
    logging.error(f"Failed to save models: {e}")