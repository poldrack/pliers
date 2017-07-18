''' Filters that operate on ImageStim inputs. '''

from pliers.stimuli import ImageStim
from .base import Filter

try:
    from skimage.color import convert_colorspace, gray2rgb, rgb2gray
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
        ``['GRAY', 'GREY', 'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ',
        'YPbPr', 'YCbCr']``.
    '''

    _log_attributes = ('source', 'target')

    def __init__(self, source, target):
        self.source = source
        self.target = target
        super(ImageFilter, self).__init__()

    def _filter(self, stim):
        data = stim.data
        if self.source in ['GRAY', 'GREY']:
            data = gray2rgb(data)
            self.source = 'RGB'
        to_gray = False
        if self.target in ['GRAY', 'GREY']:
            self.target = 'RGB'
            to_gray = True
        data = convert_colorspace(data, self.source, self.target)
        if to_gray:
            data = rgb2gray(data)
        return ImageStim(filename=stim.filename, data=data,
                         onset=stim.onset, duration=stim.duration)
