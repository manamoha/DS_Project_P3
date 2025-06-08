import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import logging
import pandas as pd
import sqlite3

logging.basicConfig(filename='pipeline_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s')

def train_model(train_generator, val_generator, train_df, val_df, test_generator=None, test_df=None):
    try:
        # Compute class weights
        conn = sqlite3.connect('./data/DogImages.db')
        train_data = pd.read_sql_query("SELECT Label FROM images WHERE Split = 'train'", conn)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_data['Label']), y=train_data['Label'])
        class_weights_dict = dict(enumerate(class_weights))
        conn.close()
        logging.info(f"Class weights: {class_weights_dict}")

        # Define VGG16 model
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
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='weighted')
        logging.info(f"VGG16 - Val Accuracy: {val_acc:.4f}, F1-Score: {val_f1:.4f}")

        # Test evaluation
        test_acc, test_f1, test_prec, test_rec, test_cm = None, None, None, None, None
        if test_generator is not None and test_df is not None:
            logging.info("Evaluating VGG16 on test set...")
            test_pred = model.predict(test_generator).argmax(axis=1)
            test_true = test_df['Label'].astype(int).values
            test_acc = accuracy_score(test_true, test_pred)
            test_f1 = f1_score(test_true, test_pred, average='weighted')
            test_prec = precision_score(test_true, test_pred, average='weighted')
            test_rec = recall_score(test_true, test_pred, average='weighted')
            test_cm = confusion_matrix(test_true, test_pred)
            logging.info(f"VGG16 - Test Accuracy: {test_acc:.4f}, F1-Score: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

            # Save confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Dalmatian', 'Golden Retriever', 'Pug', 'Rottweiler'],
                        yticklabels=['Dalmatian', 'Golden Retriever', 'Pug', 'Rottweiler'])
            plt.title('VGG16 Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('vgg16_cm_sec3.png')
            plt.close()
            logging.info("Confusion matrix saved: vgg16_cm_sec3.png")

        # Save model
        model.save('./vgg16_model.h5')
        logging.info("Model saved: ./vgg16_model.h5")

        return model, history, val_acc, val_f1, test_acc, test_f1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e