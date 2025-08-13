import os
import tempfile
import hashlib
import numpy as np
import torch
import torchaudio
from typing import Optional, Union
import threading

# Импорты из оригинального кода
from uz_tts.model import CFM
from uz_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from vocos import Vocos
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence


class TTS:

    def __init__(
        self,
        ref_audio_path: str,
        ref_text: str = "",
        model_cls=None,
        model_cfg: dict = None,
        ckpt_path: str = "",
        vocoder_name: str = "vocos",
        device: str = "auto",
        preload_models: bool = True,
        use_cache: bool = True,
        target_rms: float = 0.1,
        speed: float = 1.0,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        vocab: str=""
    ):
        """
        Инициализация голосового ассистента.
        
        Args:
            ref_audio_path: Путь к референсному аудиофайлу
            ref_text: Референсный текст (если пустой, будет транскрибирован)
            model_cls: Класс модели
            model_cfg: Конфигурация модели
            ckpt_path: Путь к чекпоинту модели
            vocoder_name: Имя вокодера ("vocos" или "bigvgan")
            device: Устройство для вычислений ("auto", "cuda", "cpu", etc.)
            preload_models: Предзагрузить модели при инициализации
            use_cache: Использовать кеширование
            target_rms: Целевой RMS для нормализации
            speed: Скорость речи
            nfe_step: Количество шагов NFE
            cfg_strength: Сила CFG
        """
        
        # Конфигурация
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.vocoder_name = vocoder_name
        self.use_cache = use_cache
        self.target_rms = target_rms
        self.speed = speed
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        
        # Константы
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        self.mel_spec_type = "vocos"
        self.ode_method = "euler"
        self.sway_sampling_coef = -1.0
        
        # Определение устройства
        if device == "auto":
            self.device = self._auto_detect_device()
        else:
            self.device = device
            
        # Инициализация состояния
        self.model = None
        self.vocoder = None
        self.asr_pipe = None
        self.processed_ref_audio = None
        self.processed_ref_text = None
        self.tokenizer = None
        self.vocab_char_map = None
        
        # Кеши
        self._audio_cache = {}
        self._text_cache = {}
        
        # Потокобезопасность
        self._lock = threading.Lock()
        self._initialized = False
        
        # Предзагрузка моделей
        if preload_models:
            self.initialize_models(model_cls, model_cfg, ckpt_path,vocab)
            
    def _auto_detect_device(self) -> str:
        """Автоматическое определение устройства"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def initialize_models(
        self, 
        model_cls=None, 
        model_cfg: dict = None, 
        ckpt_path: str = "",
        vocab: str = ""
    ):
        """Инициализация всех моделей"""
        with self._lock:
            if self._initialized:
                return
                
            print(f"Инициализация моделей на устройстве: {self.device}")
            
            # Загрузка вокодера
            self._load_vocoder()
            
            # Загрузка основной модели
            if model_cls and model_cfg and ckpt_path and vocab:
                self._load_main_model(model_cls, model_cfg, ckpt_path,vocab)
            
            # Предобработка референсного аудио
            self._preprocess_reference()
            
            self._initialized = True
            print("Инициализация завершена!")
            
    def _load_vocoder(self):
        """Загрузка вокодера"""
        if self.vocoder_name == "vocos":
            print("Загрузка Vocos...")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            
            self.vocoder = Vocos.from_hparams(config_path)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            
            from vocos.feature_extractors import EncodecFeatures
            if isinstance(self.vocoder.feature_extractor, EncodecFeatures):
                encodec_parameters = {
                    "feature_extractor.encodec." + key: value
                    for key, value in self.vocoder.feature_extractor.encodec.state_dict().items()
                }
                state_dict.update(encodec_parameters)
                
            self.vocoder.load_state_dict(state_dict)
            self.vocoder = self.vocoder.eval().to(self.device)
            
    def _load_main_model(self, model_cls, model_cfg: dict, ckpt_path: str,vocab):


        self.vocab_char_map, vocab_size = get_tokenizer(vocab, "custom")
        model_arc = model_cfg.model.arch
        # Создание модели
        self.model = CFM(
            transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=self.n_mel_channels),
            mel_spec_kwargs=dict(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mel_channels=self.n_mel_channels,
                target_sample_rate=self.target_sample_rate,
                mel_spec_type=self.mel_spec_type,
            ),
            odeint_kwargs=dict(method=self.ode_method),
            vocab_char_map=self.vocab_char_map,
        ).to(self.device)
        # Загрузка чекпоинта
        self._load_checkpoint(ckpt_path)
        try:
            self.model = torch.compile(self.model)
            print("Модель скомпилирована с помощью torch.compile().")
        except Exception as e:
            print(f"Не удалось скомпилировать модель: {e}")

    def _load_checkpoint(self, ckpt_path: str):
        """Загрузка чекпоинта модели"""
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = self.model.to(dtype)
     #   self.model = torch.compile(self.model)
        ckpt_type = ckpt_path.split(".")[-1]
        if ckpt_type == "safetensors":
            from safetensors.torch import load_file
            checkpoint = load_file(ckpt_path, device=self.device)
            checkpoint = {"ema_model_state_dict": checkpoint}
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            
        # Использование EMA весов
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        
        # Патч для обратной совместимости
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]
                
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        torch.cuda.empty_cache()
        

    def _preprocess_reference(self):
        """Предобработка референсного аудио и текста"""
        print("Предобработка референсного аудио...")
        
        # Хеширование для кеша
        with open(self.ref_audio_path, "rb") as f:
            audio_hash = hashlib.md5(f.read()).hexdigest()
            
        # Обработка аудио
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            
        aseg = AudioSegment.from_file(self.ref_audio_path)
        
        # Обрезка длинного аудио
        if len(aseg) > 12000:
            # Попытка найти тишину для обрезки
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, 
                keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 12000:
                    break
                non_silent_wave += seg
            aseg = non_silent_wave if len(non_silent_wave) <= 12000 else aseg[:12000]
            
        # Удаление тишины с краев
        aseg = self._remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(temp_path, format="wav")
        
        # Загрузка и нормализация
        audio, sr = torchaudio.load(temp_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
            
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
            
        self.processed_ref_audio = audio.to(self.device)
        
        # Обработка текста
      
        self.processed_ref_text = self.ref_text
            
        # Добавление финальной пунктуации
        if not self.processed_ref_text.endswith(". ") and not self.processed_ref_text.endswith("。"):
            if self.processed_ref_text.endswith("."):
                self.processed_ref_text += " "
            else:
                self.processed_ref_text += ". "
                
        print(f"Референсный текст: {self.processed_ref_text}")
        
        # Очистка временного файла
        os.unlink(temp_path)
        
    def _remove_silence_edges(self, audio_segment):
        """Удаление тишины с краев аудио"""
        silence_threshold = -42
        
        # Удаление тишины в начале
        non_silent_start = silence.detect_leading_silence(audio_segment, silence_threshold)
        audio_segment = audio_segment[non_silent_start:]
        
        # Удаление тишины в конце
        non_silent_end_duration = audio_segment.duration_seconds
        for ms in reversed(audio_segment):
            if ms.dBFS > silence_threshold:
                break
            non_silent_end_duration -= 0.001
            
        return audio_segment[:int(non_silent_end_duration * 1000)]
        
    def generate_speech(
        self, 
        text: str, 
        speed: Optional[float] = None,
        return_numpy: bool = True
    ) -> Union[tuple, np.ndarray]:
        """
        Генерация речи для заданного текста.
        
        Args:
            text: Текст для синтеза
            speed: Скорость речи (если None, используется значение по умолчанию)
            return_numpy: Возвращать numpy array или torch tensor
            
        Returns:
            Если return_numpy=True: (audio_array, sample_rate)
            Если return_numpy=False: (audio_tensor, sample_rate)
        """
        
        if not self._initialized:
            raise RuntimeError("Модели не инициализированы! Вызовите initialize_models() сначала.")
            
        if not text.strip():
            raise ValueError("Текст не может быть пустым!")
            
        current_speed = speed if speed is not None else self.speed
        
        # Корректировка скорости для очень короткого текста
        if len(text.encode("utf-8")) < 10:
            current_speed = 0.3
            
        with self._lock:
            # Подготовка текста
            if len(self.processed_ref_text[-1].encode("utf-8")) == 1:
                ref_text_adjusted = self.processed_ref_text + " "
            else:
                ref_text_adjusted = self.processed_ref_text
                
            full_text = ref_text_adjusted + text
            text_list = [full_text]
            final_text_list = convert_char_to_pinyin(text_list)
            
            # Расчет длительности
            ref_audio_len = self.processed_ref_audio.shape[-1] // self.hop_length
            ref_text_len = len(ref_text_adjusted.encode("utf-8"))
            gen_text_len = len(text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / current_speed)
            
            # Генерация
            with torch.inference_mode():
                generated, _ = self.model.sample(
                    cond=self.processed_ref_audio,
                    text=final_text_list,
                    duration=duration,
                    steps=self.nfe_step,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=self.sway_sampling_coef,
                )
                
                generated = generated.to(torch.float32)
                generated = generated[:, ref_audio_len:, :]
                generated = generated.permute(0, 2, 1)
                
                # Декодирование через вокодер
                if self.mel_spec_type == "vocos":
                    generated_wave = self.vocoder.decode(generated)
                else:  # bigvgan
                    generated_wave = self.vocoder(generated)
                    
                # Нормализация по RMS
                rms = torch.sqrt(torch.mean(torch.square(self.processed_ref_audio)))
                if rms < self.target_rms:
                    generated_wave = generated_wave * rms / self.target_rms
                    
                generated_wave = generated_wave.squeeze()
                
                if return_numpy:
                    return generated_wave.cpu().numpy(), self.target_sample_rate
                else:
                    return generated_wave, self.target_sample_rate
                    
    def save_audio(self, audio_data: np.ndarray, output_path: str, sample_rate: int = None):
        """Сохранение аудио в файл"""
        if sample_rate is None:
            sample_rate = self.target_sample_rate
            
        # Конвертация в torch tensor если нужно
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            audio_tensor = audio_data.unsqueeze(0) if audio_data.dim() == 1 else audio_data
            
        torchaudio.save(output_path, audio_tensor, sample_rate)
        
    def get_info(self) -> dict:
        """Получение информации о конфигурации"""
        return {
            "device": self.device,
            "target_sample_rate": self.target_sample_rate,
            "vocoder": self.vocoder_name,
            "initialized": self._initialized,
            "ref_text": self.processed_ref_text,
            "speed": self.speed,
            "nfe_step": self.nfe_step,
            "cfg_strength": self.cfg_strength
        }
        
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'vocoder') and self.vocoder is not None:
            del self.vocoder
        if hasattr(self, 'asr_pipe') and self.asr_pipe is not None:
            del self.asr_pipe
        torch.cuda.empty_cache()




