from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import librosa
import gradio as gr



model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=True)

audio_path = "/home/team1/Sakthivel_f22/onnx/Bria.wav"
def transcription(audio_path):
    # Load the audio file
    audio, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
    # Transcribe the audio
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, max_length=448)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return(transcription[0])


gr.Interface(
    fn=transcription,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.TextArea(type="text")
).launch()