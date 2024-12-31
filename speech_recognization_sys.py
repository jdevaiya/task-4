from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

def transcribe_with_wav2vec(audio_path):
    # Load the pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Load audio file
    speech, rate = librosa.load(audio_path, sr=16000)

    # Preprocess audio
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    
    # Perform inference
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode output
    transcription = processor.decode(predicted_ids[0])
    print(f"Transcription: {transcription}")
    return transcription

# Test the function
transcribe_with_wav2vec("your_audio_file.wav")
