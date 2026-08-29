# Spoken Language Identification (14 Languages)

## Objective
Spoken Language Identification (LID) recognizes the language of a given spoken speech utterance. This project classifies **14 languages** across Dravidian, Indo-Aryan, Romance, Germanic, and Sino-Tibetan families:
- **International / Benchmark Languages**: English (🇬🇧), Mandarin (🇨🇳), French (🇫🇷), German (🇩🇪), Spanish (🇪🇸)
- **Indian Languages**: Hindi (🇮🇳), Tamil (🇮🇳), Telugu (🇮🇳), Kannada (🇮🇳), Malayalam (🇮🇳), Bengali (🇮🇳), Marathi (🇮🇳), Gujarati (🇮🇳), Punjabi (🇮🇳)

We implement a deep LSTM neural network using 64-dimensional Mel-Frequency Cepstral Coefficients (MFCC) extracted from 16kHz audio.

## Supported Languages & Mapping
- `English`: 0
- `Hindi`: 1
- `Mandarin`: 2
- `Tamil`: 3
- `Telugu`: 4
- `Kannada`: 5
- `Malayalam`: 6
- `Bengali`: 7
- `Marathi`: 8
- `Gujarati`: 9
- `Punjabi`: 10
- `French`: 11
- `German`: 12
- `Spanish`: 13

## Sample length
The full audio files are ∼ 10 minutes long which might be too long to train an RNN. Multiple 10 seconds samples are created from every utterance and the same label as the original utterance are assigned to them. The choice of sequence length can be changed to experiment with samples of different length.

## Audio Format
The wav files have 16KHz sampling rate, single channel, and 16-bit Signed Integer PCM encoding.

## Notes about the code
The code has been divided into 6 blocks. Kindly refer to the following notes to comment/uncomment the blocks as needed

- The code in Block 1 is used to extract the mfcc features provided and write them into a dataset “mfcc_dataset.hdf5”. This part of the code can be commented out if the hdf5 file already exists.

- The code in Block 2 is used to read the “mfcc_dataset.hdf5” dataset. Do not comment it out.

- The code in Block 3 is used to train the model. Comment it out after the model has been trained and saved by the name “sld.hdf5”

- The code in Block 4 sets up the inference mode.

- The code in Block 5 runs the streaming model in inference mode by predicting the label for a single random sequence from the validation dataset.

- The code in Block 6 runs the streaming model in inference mode by predicting the the labels for all the sequences in the validation dataset. Comment this out since it can take a long time to run.
