"""Model construction shared by CLI and UI (no dependency on either)."""


def load_recognizer(config):
    if config.ocr_engine == "none":
        return None
    if config.ocr_engine == "crnn":
        from .crnn import CRNNRecognizer

        return CRNNRecognizer(config.ocr_weights, device=config.device)
    from .ocr import VietnamesePlateRecognizer

    return VietnamesePlateRecognizer(config.ocr_weights, config.ocr_config)
