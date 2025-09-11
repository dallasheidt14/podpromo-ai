# backend/services/title_gen.py
# DEPRECATED shim: use title_service.generate_titles

from .title_service import generate_titles, normalize_platform
__all__ = ["generate_titles", "normalize_platform"]