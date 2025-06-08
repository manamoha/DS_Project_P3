import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight  # Added missing import
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    filename='vgg16_evaluation_log.log',
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

# Data generators
base_dir = r"D:/university/sem 8/Data/Final Project/Project_P3_810100207_810100247_810102/"  # Adjust if needed
def create_data_generator(datagen, df_subset, batch_size=16, target_size=(224, 224), shuffle=True):
    return datagen.flow_from_dataframe(
        dataframe=df_subset,
        directory=base_dir,
        x_col='Image_File_Path',
        y_col='Label_str',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=shuffle,
        validate_filenames=True
    )

train_generator = create_data_generator(train_datagen, train_df, batch_size=16, shuffle=True)
val_generator = create_data_generator(val_test_datagen, val_df, batch_size=16, shuffle=False)
test_generator = create_data_generator(val_test_datagen, test_df, batch_size=16, shuffle=False)
logging.info("Data generators configured")

# VGG16 Model
def train_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-5]:
        layer.trainable = False
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

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    logging.info("Training VGG16...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=(len(train_df) + 15) // 16,
        validation_steps=(len(val_df) + 15) // 16,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )

    # Validation monitoring
    val_pred = model.predict(val_generator).argmax(axis=1)
    val_true = val_df['Label'].astype(int).values
    acc = accuracy_score(val_true, val_pred)
    f1 = f1_score(val_true, val_pred, average='weighted')
    logging.info(f"VGG16 - Val Accuracy: {acc:.4f}, Val F1-Score: {f1:.4f}")

    return model, history, acc, f1

# Try loading saved model or retrain
try:
    model = load_model('vgg16_model.h5')
    logging.info("Loaded saved VGG16 model")
    # Validation check for loaded model
    val_pred = model.predict(val_generator).argmax(axis=1)
    val_true = val_df['Label'].astype(int).values
    acc = accuracy_score(val_true, val_pred)
    f1 = f1_score(val_true, val_pred, average='weighted')
    history = None
except Exception as e:
    logging.info(f"Failed to load VGG16 model: {e}. Retraining...")
    model, history, acc, f1 = train_vgg16()
    model.save("vgg16_model.h5")
    logging.info("VGG16 model saved: vgg16_model.h5")

# Print validation results
print("\nVGG16 Validation Monitoring (Not Final Evaluation):")
print(f"Validation Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")
logging.info(f"VGG16 Monitoring: Validation Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

# Evaluate on test set
logging.info("Evaluating VGG16 on test set...")
test_pred = model.predict(test_generator).argmax(axis=1)
test_true = test_df['Label'].astype(int).values
acc_test = accuracy_score(test_true, test_pred)
f1_test = f1_score(test_true, test_pred, average='weighted')
prec_test = precision_score(test_true, test_pred, average='weighted')
rec_test = recall_score(test_true, test_pred, average='weighted')
cm_test = confusion_matrix(test_true, test_pred)

# Print test results
print("\nVGG16 Test Set Evaluation:")
print(f"Accuracy: {acc_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")
print(f"Precision: {prec_test:.4f}")
print(f"Recall: {rec_test:.4f}")
logging.info(f"VGG16 - Test Accuracy: {acc_test:.4f}, F1-Score: {f1_test:.4f}, Precision: {prec_test:.4f}, Recall: {rec_test:.4f}")

# Plot and save confusion matrix
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dalmatian', 'Golden Retriever', 'Pug', 'Rottweiler'], 
                yticklabels=['Dalmatian', 'Golden Retriever', 'Pug', 'Rottweiler'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()
    logging.info(f"Confusion matrix saved: {filename}")

plot_confusion_matrix(cm_test, "VGG16 Confusion Matrix", "vgg16_cm_sec2.png")