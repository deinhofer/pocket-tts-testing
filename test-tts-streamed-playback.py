from pocket_tts import TTSModel
import scipy.io.wavfile
import sounddevice as sd
import numpy as np

tts_model = TTSModel.load_model(language="german_24l",eos_threshold=-10.0,lsd_decode_steps=1,quantize=True)
voice_state = tts_model.get_state_for_audio_prompt(
    "juergen"  # One of the pre-made voices, see above
    # You can also use any voice file you have locally or from Hugging Face:
    # "./some_audio.wav"
    # or "hf://kyutai/tts-voices/expresso/ex01-ex02_default_001_channel2_198s.wav"
)

# Stream generation with buffering
chunk_buffer = []
stream_started = False
stream = sd.OutputStream(samplerate=tts_model.sample_rate,channels=1)
buffer_threshold = 2  # Number of chunks to buffer before starting playback

text = input("Enter text to speak: ")
#audio = tts_model.generate_audio(voice_state, "Heute ist es sehr wechselhaft, es könnte regnen oder die Sonne könnte scheinen.")
# Audio is a 1D torch tensor containing PCM data.
#scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())

for chunk in tts_model.generate_audio_stream(voice_state, text):
    # Process each chunk as it's generated
    print(f"Generated chunk: {chunk.shape[0]} samples")
    
    # Buffer the first 2 chunks before starting playback
    if not stream_started:
        chunk_buffer.append(chunk)
        print(f"Buffered chunk {len(chunk_buffer)} of {buffer_threshold}")
        if len(chunk_buffer) == buffer_threshold:
            # Start stream and write buffered chunks
            stream.start()
            stream_started = True
            for buffered_chunk in chunk_buffer:
                stream.write(buffered_chunk.numpy())
    else:
        # Stream is already running, write chunk directly
        print("Streaming chunk directly")
        stream.write(chunk.numpy())

stream.stop()
stream.close()
    