import numpy as np
import librosa
import os

LANGUAGES = [
    "English", "Hindi", "Mandarin", "Tamil", "Telugu",
    "Kannada", "Malayalam", "Bengali", "Marathi", "Gujarati", "Punjabi",
    "French", "German", "Spanish"
]

def extract_mfcc_from_audio(audio_path, target_sr=16000, num_mfcc=64, target_length=1000):
    """
    Extracts 64-dimensional MFCC features from an audio file (.wav)
    conforming to 16kHz sample rate and sequence length of 1000 frames (10 seconds).
    """
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    hop_length = 160
    n_fft = 512
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    
    if len(mfcc) < target_length:
        pad_width = target_length - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='wrap')
    else:
        mfcc = mfcc[:target_length]
        
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    return mfcc.astype(np.float32)

def generate_sample_wavs(output_dir="./sample_audio"):
    """
    Generates synthetic sample .wav audio files for all 14 languages
    """
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    import soundfile as sf
    
    acoustic_profiles = {
        "english": (
            0.5 * np.sin(2 * np.pi * 500 * t) * (np.sin(2 * np.pi * 2 * t) > 0) +
            0.3 * np.sin(2 * np.pi * 1500 * t) +
            0.2 * np.random.normal(0, 0.08, len(t))
        ),
        "hindi": (
            0.5 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 4 * t) > -0.2) +
            0.4 * np.sin(2 * np.pi * 2200 * t) +
            0.2 * np.random.normal(0, 0.08, len(t))
        ),
        "mandarin": (
            0.6 * np.sin(2 * np.pi * (250 + 100 * np.sin(2 * np.pi * 0.8 * t)) * t) +
            0.3 * np.sin(2 * np.pi * 1800 * t) +
            0.2 * np.random.normal(0, 0.08, len(t))
        ),
        "tamil": (
            0.5 * np.sin(2 * np.pi * 420 * t) * (np.sin(2 * np.pi * 5 * t) > -0.1) +
            0.4 * np.sin(2 * np.pi * 2600 * t) +
            0.3 * np.sin(2 * np.pi * 3800 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "telugu": (
            0.55 * np.sin(2 * np.pi * 380 * t) * np.cos(2 * np.pi * 1.5 * t) +
            0.35 * np.sin(2 * np.pi * 1600 * t) +
            0.25 * np.sin(2 * np.pi * 2900 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "kannada": (
            0.5 * np.sin(2 * np.pi * 400 * t) +
            0.4 * np.sin(2 * np.pi * 1900 * t) * (np.sin(2 * np.pi * 3.5 * t) > 0.3) +
            0.3 * np.sin(2 * np.pi * 3100 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "malayalam": (
            0.5 * np.sin(2 * np.pi * 450 * t) * np.sin(2 * np.pi * 3 * t) +
            0.4 * np.sin(2 * np.pi * 2400 * t) +
            0.35 * np.sin(2 * np.pi * 3400 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "bengali": (
            0.6 * np.sin(2 * np.pi * 280 * t) * (np.sin(2 * np.pi * 1.8 * t) > -0.3) +
            0.35 * np.sin(2 * np.pi * 1300 * t) +
            0.25 * np.sin(2 * np.pi * 2700 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "marathi": (
            0.5 * np.sin(2 * np.pi * 350 * t) * (np.sin(2 * np.pi * 3.2 * t) > 0) +
            0.4 * np.sin(2 * np.pi * 2100 * t) +
            0.35 * np.sin(2 * np.pi * 4200 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "gujarati": (
            0.5 * np.sin(2 * np.pi * 320 * t) +
            0.4 * np.sin(2 * np.pi * 1700 * t) +
            0.3 * np.random.normal(0, 0.18, len(t))
        ),
        "punjabi": (
            0.6 * np.sin(2 * np.pi * (300 + 80 * np.sin(2 * np.pi * 1.5 * t)) * t) +
            0.35 * np.sin(2 * np.pi * 2000 * t) +
            0.2 * np.random.normal(0, 0.08, len(t))
        ),
        "french": (
            0.5 * np.sin(2 * np.pi * 480 * t) * (np.sin(2 * np.pi * 2.8 * t) > -0.2) +
            0.4 * np.sin(2 * np.pi * 1650 * t) +
            0.3 * np.sin(2 * np.pi * 3200 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        ),
        "german": (
            0.6 * np.sin(2 * np.pi * 340 * t) * (np.sin(2 * np.pi * 3.6 * t) > 0.1) +
            0.45 * np.sin(2 * np.pi * 1850 * t) +
            0.35 * np.sin(2 * np.pi * 3600 * t) +
            0.2 * np.random.normal(0, 0.08, len(t))
        ),
        "spanish": (
            0.55 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 4.2 * t) > -0.1) +
            0.4 * np.sin(2 * np.pi * 2300 * t) +
            0.25 * np.sin(2 * np.pi * 3900 * t) +
            0.15 * np.random.normal(0, 0.08, len(t))
        )
    }
    
    for lang, audio_signal in acoustic_profiles.items():
        # Normalize peak audio
        audio_signal = audio_signal / np.max(np.abs(audio_signal) + 1e-6) * 0.9
        sf.write(os.path.join(output_dir, f"sample_{lang}.wav"), audio_signal, sr)
        
    print(f"Generated sample audio for {len(acoustic_profiles)} languages in {output_dir}/")

if __name__ == "__main__":
    generate_sample_wavs()
