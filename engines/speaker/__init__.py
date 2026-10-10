"""Speaker diarization engine package (wave D-3, CAM++ + FSMN-VAD)."""


from .campp_engine import CAMPPEngine, run_diarization


#: Class resolved by `engines.get_engine("speaker")`.
engine_class = CAMPPEngine

__all__ = ["CAMPPEngine", "run_diarization", "engine_class"]
