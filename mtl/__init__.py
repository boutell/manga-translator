"""Self-hosted manga translation pipeline.

Detect -> OCR (Japanese) -> translate -> render English over an opaque box.
Everything runs on-device; no network calls except one-time model downloads.
"""
