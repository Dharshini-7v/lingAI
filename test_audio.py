import sys
import os
import numpy as np
import tensorflow as tf
from audio_utils import extract_mfcc_from_audio, LANGUAGES

def predict_language(audio_path, model_path="sld.keras"):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_path}' not found. Please train it first using python language_identification.py.")
        return
        
    print(f"\n[INFO] Loading audio file: {audio_path}")
    mfcc_feat = extract_mfcc_from_audio(audio_path)
    print(f"[INFO] Extracted MFCC features: shape {mfcc_feat.shape} (1000 frames x 64 coefficients)")
    
    print(f"[INFO] Loading trained {len(LANGUAGES)}-language model...")
    model = tf.keras.models.load_model(model_path)
    
    input_tensor = np.expand_dims(mfcc_feat, axis=0) # (1, 1000, 64)
    preds = model.predict(input_tensor, verbose=0)[0] # (1000, 11)
    
    avg_probs = np.mean(preds, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    
    print("\n" + "=" * 56)
    print("      MULTILINGUAL SPOKEN LANGUAGE IDENTIFICATION")
    print("=" * 56)
    print(f" Predicted Language: {LANGUAGES[pred_idx].upper()}")
    print(f" Confidence Score:   {avg_probs[pred_idx] * 100:.2f}%")
    print("-" * 56)
    print(" Probability Distribution:")
    for lang, prob in zip(LANGUAGES, avg_probs):
        bar_len = int(prob * 25)
        bar = "#" * bar_len
        print(f"   {lang:<12}: {prob*100:6.2f}%  [{bar:<25}]")
    print("=" * 56 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_audio = sys.argv[1]
    else:
        target_audio = "./sample_audio/sample_tamil.wav"
        print(f"No file specified. Defaulting to sample: {target_audio}")
        
    predict_language(target_audio)
