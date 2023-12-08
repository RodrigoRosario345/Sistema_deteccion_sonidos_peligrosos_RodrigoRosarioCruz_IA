import pyaudio
import numpy as np
import librosa
import torch
import esc_config as config
from model.htsat import HTSAT_Swin_Transformer

#Modelo entrenado con el dataset de sonidos peligrosos
model_path = 'workspace/results/exp_htsat_dataset_sonidos_peligrosos/checkpoint/lightning_logs/version_4/checkpoints/l-epoch=49-acc=0.905.ckpt'

#Clase para clasificar los audios
class Audio_Classification:
    def __init__(self, model_path, config):
        super().__init__()
        self.device = torch.device('cuda')
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
        ckpt = torch.load(model_path, map_location="cpu")
        temp_ckpt = {}
        for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt['state_dict'][key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()


    def predict(self, waveform):
        if waveform.size > 0:
            with torch.no_grad():
                x = torch.from_numpy(waveform).float().to(self.device)
                output_dict = self.sed_model(x[None, :], None, True)
                pred = output_dict['clipwise_output']
                pred_post = pred[0].detach().cpu().numpy()
                pred_label = np.argmax(pred_post)
                pred_prob = np.max(pred_post)
            return pred_label, pred_prob



#Clase para procesar el audio en tiempo real
class RealTimeAudioProcessor:
    def __init__(self, model, silence_threshold=0.03, format=pyaudio.paFloat32, channels=1, rate=32000, chunk_size=1024, device_index=None):
        self.model = model
        self.silence_threshold = silence_threshold  # Threshold to determine if audio is silence
        self.audio_format = format
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.audio_interface = pyaudio.PyAudio()

    def start_stream(self):
        stream = self.audio_interface.open(format=self.audio_format, channels=self.channels,
                                           rate=self.rate, input=True, frames_per_buffer=self.chunk_size,
                                           input_device_index=self.device_index)
        return stream

    def is_silence(self, audio_data):
        # Check if the audio is quieter than the silence threshold
        rms = np.sqrt(np.mean(np.square(audio_data)))
        print(f"RMS: {rms}")
        return rms < self.silence_threshold

    def process_stream(self, stream):
        print("Starting audio stream...")
        audio_buffer = []
        while True:
            data = stream.read(self.chunk_size)
            numpy_data = np.frombuffer(data, dtype=np.float32)  # Convertir a numpy array
            if numpy_data.size > 0:  # Verificar que el array no esté vacío
                audio_buffer.append(numpy_data)
            if len(audio_buffer) == self.rate // self.chunk_size * 5:  # 5 segundos de audio
                if any(numpy_data.size > 0 for numpy_data in audio_buffer):  # Verificar que hay datos no vacíos
                    concatenated_data = np.concatenate(audio_buffer)
                    if not self.is_silence(concatenated_data):
                        self.predict(concatenated_data)
                    else:
                        print("Detected silence, skipping prediction.")
                audio_buffer = []


    def predict(self, audio_data):
        # Additional preprocessing if needed
        pred_label, pred_prob = self.model.predict(audio_data)
        print(f"Predicted: {pred_label} with probability {pred_prob}")

    def run(self):
        stream = self.start_stream()
        self.process_stream(stream)

#Instanciamos la clase
audiocls = Audio_Classification(model_path, config)

#Ejecutamos el programa y clasificamos el audio en tiempo real
if __name__ == "__main__":
    processor = RealTimeAudioProcessor(audiocls)
    processor.run()
