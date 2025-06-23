"""
services/translator.py - Translation services for JazzBot
"""

from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
from translatepy import Translator
from indicnlp import common

from config import config
from utils.logger import logger


class TranslationService:
    """Translation service for Roman Urdu support"""
    
    def __init__(self):
        # Configure Indic NLP resources path
        common.set_resources_path(config.INDIC_NLP_RESOURCES_PATH)
        
        # Initialize the translator
        self.translator = Translator()
    
    def translate_to_roman_urdu(self, text: str) -> str:
        """Translate text to Urdu and convert it to Roman Urdu using Indic NLP."""
        try:
            # Translate text to Urdu
            translated = self.translator.translate(text, "Urdu")
            urdu_text = translated.result

            # Transliterate Urdu script to Roman script
            roman_urdu = ItransTransliterator.to_itrans(urdu_text, "ur")
            return roman_urdu
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Fallback to the original text


# Global translator instance
translator_service = TranslationService()