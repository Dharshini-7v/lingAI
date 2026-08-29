import numpy as np
import glob
import os
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras import optimizers, callbacks
from sklearn.model_selection import train_test_split

LANGUAGES = [
    "English",    # 0
    "Hindi",      # 1
    "Mandarin",   # 2
    "Tamil",      # 3
    "Telugu",     # 4
    "Kannada",    # 5
    "Malayalam",  # 6
    "Bengali",    # 7
    "Marathi",    # 8
    "Gujarati",   # 9
    "Punjabi",    # 10
    "French",     # 11
    "German",     # 12
    "Spanish"     # 13
]

num_classes = len(LANGUAGES)

def language_name(index):
    if 0 <= index < len(LANGUAGES):
        return LANGUAGES[index]
    return f"Unknown ({index})"

# ---------------------------BLOCK 1------------------------------------
# Extract MFCC features from dataset
codePath = './train/'
num_mfcc_features = 64
sequence_length = 1000

print(f"[BLOCK 1] Generating multilingual dataset for {num_classes} languages...")
from generate_dataset import create_dataset_from_speech
create_dataset_from_speech(output_hdf5="mfcc_dataset.hdf5", train_dir=codePath)
# ---------------------------------------------------------------


# --------------------------BLOCK 2-------------------------------------
# Load MFCC Dataset
print("\n[BLOCK 2] Loading dataset from 'mfcc_dataset.hdf5'...")
with h5py.File("mfcc_dataset.hdf5", 'r') as hf:
    X_train = hf['X_train'][:]
    Y_train = hf['Y_train'][:]
    X_val = hf['X_val'][:]
    Y_val = hf['Y_val'][:]

print(f"Dataset Loaded Successfully ({num_classes} Languages):")
print(f"  X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"  X_val shape:   {X_val.shape}, Y_val shape:   {Y_val.shape}")
# ---------------------------------------------------------------


# ---------------------------BLOCK 3------------------------------------
# Setting up the model for training
optimizer = optimizers.Adam(learning_rate=0.002)
main_input = Input(shape=(sequence_length, 64), name='main_input')

# Modern Deep LSTM architecture
layer1 = LSTM(64, return_sequences=True, name='layer1')(main_input)
layer2 = LSTM(32, return_sequences=True, name='layer2')(layer1)
drop = Dropout(0.15, name='dropout')(layer2)
layer3 = Dense(100, activation='tanh', name='layer3')(drop)
rnn_output = Dense(num_classes, activation='softmax', name='rnn_output')(layer3)

model = Model(inputs=main_input, outputs=rnn_output)
print(f'\n[BLOCK 3] Compiling {num_classes}-Language Model...')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=4,
    min_lr=1e-5,
    verbose=1
)

epochs = 28
print(f"\n[BLOCK 3] Training model for {epochs} epochs on {len(X_train)} sequences...")
history = model.fit(
    X_train, Y_train,
    batch_size=16,
    epochs=epochs,
    validation_data=(X_val, Y_val),
    callbacks=[lr_scheduler],
    shuffle=True,
    verbose=1
)
model.save('sld.keras')
model.save_weights('sld.weights.h5')
print("[BLOCK 3] Model saved to 'sld.keras' and weights saved to 'sld.weights.h5'")
# ---------------------------------------------------------------


# --------------------------BLOCK 4-------------------------------------
# Inference Mode Setup (Streaming stateful LSTM)
print("\n[BLOCK 4] Setting up streaming inference model...")
streaming_input = Input(name='streaming_input', batch_shape=(1, 1, 64))
pred_layer1 = LSTM(64, return_sequences=True, name='layer1', stateful=True)(streaming_input)
pred_layer2 = LSTM(32, return_sequences=True, name='layer2')(pred_layer1)
pred_drop = Dropout(0.15, name='dropout')(pred_layer2)
pred_layer3 = Dense(100, activation='tanh', name='layer3')(pred_drop)
pred_output = Dense(num_classes, activation='softmax', name='rnn_output')(pred_layer3)
streaming_model = Model(inputs=streaming_input, outputs=pred_output)
streaming_model.load_weights('sld.weights.h5')
print("[BLOCK 4] Streaming model loaded successfully.")
# ---------------------------------------------------------------


# ---------------------------BLOCK 5------------------------------------
print("\n[BLOCK 5] Testing inference on a random validation sample...")
pred_index = np.random.randint(0, len(X_val))
true_lang_idx = int(np.argmax(Y_val[pred_index, 0, :]))
test_seq = np.expand_dims(X_val[pred_index], axis=0)
pred_probs = model.predict(test_seq, verbose=0)[0]
pred_lang_idx = int(np.argmax(np.mean(pred_probs, axis=0)))
confidence = np.mean(pred_probs, axis=0)[pred_lang_idx] * 100

print("=" * 60)
print(f" True Language:      {language_name(true_lang_idx)}")
print(f" Predicted Language: {language_name(pred_lang_idx)} ({confidence:.2f}% confidence)")
print("=" * 60)
# ---------------------------------------------------------------


# ---------------------------BLOCK 6------------------------------------
print("\n[BLOCK 6] Evaluating validation dataset accuracy...")
val_preds = model.predict(X_val, verbose=1)
predicted_classes = np.argmax(val_preds, axis=-1).flatten()
true_classes = np.argmax(Y_val, axis=-1).flatten()

overall_acc = np.mean(predicted_classes == true_classes) * 100
print("\n" + "=" * 60)
print(f" OVERALL VALIDATION ACCURACY: {overall_acc:.2f}%")
print("=" * 60)
print(" Breakdown by Language Class:")
for i, lang in enumerate(LANGUAGES):
    correct_frames = np.sum((predicted_classes == i) & (true_classes == i))
    total_true_frames = np.sum(true_classes == i)
    total_pred_frames = np.sum(predicted_classes == i)
    acc = (correct_frames / total_true_frames * 100) if total_true_frames > 0 else 0.0
    print(f"   {lang:<12}: {acc:5.1f}% accuracy ({total_pred_frames} pred, {total_true_frames} true)")
print("=" * 60)
print(f"\n[SUCCESS] 14-Language Spoken Identification pipeline completed successfully!\n")
