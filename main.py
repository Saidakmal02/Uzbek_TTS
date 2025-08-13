from omegaconf import OmegaConf
from hydra.utils import get_class
from tts import TTS
if __name__ == "__main__":
    # Инициализация ассистента
    model_cfg=OmegaConf.load('config/UZTTS_conf.yaml')
    print(model_cfg)
    tts = TTS(
        ref_audio_path="test_data/test_erkak.wav",
        ref_text="Jizzax kollejlarida infraqizil aniqlagichli turniketlar o'rnatilmoqda.",  # Будет автоматически транскрибирован
        # ref_audio_path="test_data/test_ayol.wav",
        # ref_text="Ba'zan bunga noto'g'ri o'sgan tishlar xalaqit beradi."
        model_cfg=model_cfg,
        model_cls = get_class(f"uz_tts.model.{model_cfg.model.backbone}"),

        vocab='config/uz_vocab.txt',
        ckpt_path="ckpts/UZ.safetensors",
        device="auto",
        preload_models=True,
        speed=1
    )
    
    # Генерация речи
    audio, sr = tts.generate_speech("Salom, taklifingiz meni qiziqtirdi.")
    tts.save_audio(audio, "output1.ogg")
