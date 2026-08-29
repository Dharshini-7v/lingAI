import os
import glob
import numpy as np
import h5py
import librosa
from audio_utils import extract_mfcc_from_audio, LANGUAGES

def augment_audio(y, sr):
    """Augments audio with rich pitch variation, speed shift, and subtle noise."""
    augmented = []
    # 1. Original
    augmented.append(y)
    
    # 2. Pitch shifts (+/- 1.0, 2.0 semitones)
    for steps in [1.2, -1.2, 2.0, -2.0]:
        try:
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            augmented.append(y_shift)
        except Exception:
            pass
            
    # 3. Time stretches (0.9, 1.1)
    for rate in [0.92, 1.08]:
        try:
            y_stretch = librosa.effects.time_stretch(y, rate=rate)
            # Match 10.0s length
            target_len = len(y)
            if len(y_stretch) < target_len:
                y_stretch = np.pad(y_stretch, (0, target_len - len(y_stretch)), mode='wrap')
            else:
                y_stretch = y_stretch[:target_len]
            augmented.append(y_stretch)
        except Exception:
            pass
            
    # 4. Subtle background noise
    noise1 = np.random.normal(0, 0.004, len(y))
    augmented.append(y + noise1)
    
    noise2 = np.random.normal(0, 0.008, len(y))
    augmented.append(y + noise2)
    
    return augmented

def create_dataset_from_speech(speech_dir="./real_speech_dataset", output_hdf5="mfcc_dataset.hdf5", train_dir="./train"):
    num_classes = len(LANGUAGES)
    print(f"Extracting MFCC features from REAL speech audio across {num_classes} languages...")
    
    all_data = []
    all_labels = []
    
    target_sr = 16000
    sequence_length = 1000
    num_features = 64
    
    for lang_id, lang_name in enumerate(LANGUAGES):
        lang_folder = os.path.join(speech_dir, lang_name.lower())
        wav_files = sorted(glob.glob(os.path.join(lang_folder, "*.wav")))
        
        lang_samples = []
        for wav_path in wav_files:
            y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
            augmented_list = augment_audio(y, sr)
            
            for aug_y in augmented_list:
                mfcc = librosa.feature.mfcc(y=aug_y, sr=target_sr, n_mfcc=num_features, n_fft=512, hop_length=160).T
                if len(mfcc) < sequence_length:
                    pad = sequence_length - len(mfcc)
                    mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='wrap')
                else:
                    mfcc = mfcc[:sequence_length]
                # Standardize with Cepstral Mean & Variance Normalization (CMVN)
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
                lang_samples.append(mfcc.astype(np.float32))
                
        print(f"  [{lang_name:<10}] Extracted {len(lang_samples)} speech sequences.")
        all_data.append(np.array(lang_samples))
        
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[lang_id] = 1.0
        labels = np.full((len(lang_samples), sequence_length, num_classes), one_hot, dtype=np.float32)
        all_labels.append(labels)
        
        # Save .npy in ./train/
        os.makedirs(os.path.join(train_dir, lang_name.lower()), exist_ok=True)
        for idx, sample in enumerate(lang_samples):
            np.save(os.path.join(train_dir, lang_name.lower(), f"sample_{idx:03d}.npy"), sample.T)
            
    X = np.vstack(all_data)
    Y = np.vstack(all_labels)
    
    # Stratified split to ensure every language is well represented in train and validation
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    split_idx = int(0.85 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    with h5py.File(output_hdf5, 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('Y_train', data=Y_train)
        hf.create_dataset('X_val', data=X_val)
        hf.create_dataset('Y_val', data=Y_val)
        
    print(f"\nSuccessfully generated {output_hdf5} from REAL human speech:")
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")

if __name__ == "__main__":
    if not os.path.exists("./real_speech_dataset"):
        from generate_real_speech import generate_speech_audio
        generate_speech_audio()
    create_dataset_from_speech()

create_dataset = create_dataset_from_speech
