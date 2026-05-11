from pocket_tts import TTSModel
import scipy.io.wavfile

# Load the model
tts_model = TTSModel.load_model()

# Get voice state from an audio file
voice_state = tts_model.get_state_for_audio_prompt(
    "vera"
)

# Generate audio
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")

# Save to file
scipy.io.wavfile.write("casual.wav", tts_model.sample_rate, audio.numpy())