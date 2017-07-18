''' Filter hierarchy. '''

from .image import ColorSpaceFilter
from .text import WordStemmingFilter

__all__ = [
    'ColorSpaceFilter',
    'WordStemmingFilter'
]
