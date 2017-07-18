''' Filters that operate on ImageStim inputs. '''

from pliers.stimuli import ImageStim
from .base import Filter

try:
    from skimage.color import convert_colorspace
except ImportError:
    pass


class ImageFilter(Filter):

    ''' Base class for all TextFilters. '''

    _input_type = ImageStim


class ColorSpaceFilter(ImageFilter):

    ''' Converts an image from one color space to another.
    Args:
        source (str): source color space.
        target (str): target color space.

        Valid color space strings are
        ``['RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr']``.
    '''

    _log_attributes = ('source', 'target')

    def __init__(self, source, target):
        self.source = source
        self.target = target
        super(ImageFilter, self).__init__()

    def _filter(self, stim):
        data = convert_colorspace(stim.data, self.source, self.target)
        return ImageStim(filename=stim.filename, data=data,
                         onset=stim.onset, duration=stim.duration)
