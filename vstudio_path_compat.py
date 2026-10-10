"""Load SentencePiece model bytes through Python's Unicode-safe file APIs."""
import sys
from pathlib import Path


def configure_unicode_model_paths():
    if sys.platform != 'win32': return
    import sentencepiece
    cls = sentencepiece.SentencePieceProcessor
    if getattr(cls, '_vstudio_unicode_paths', False): return
    def load_file(self, filename):
        # The native Windows reader treats UTF-8 filenames as ANSI. Loading the
        # same serialized bytes avoids that conversion without copying models.
        return self.LoadFromSerializedProto(Path(filename).read_bytes())
    cls.LoadFromFile = load_file
    cls.load_from_file = load_file
    cls._vstudio_unicode_paths = True
