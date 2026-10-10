"""Speech ASR engine package (wave D-4): SenseVoice / Paraformer + CT punctuation."""


from .asr_engine import SpeechASREngine, run_speech_asr


#: Class resolved by `engines.get_engine("speech_asr")`.
engine_class = SpeechASREngine

__all__ = ["SpeechASREngine", "run_speech_asr", "engine_class"]
