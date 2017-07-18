from os.path import join
from .utils import get_test_data_path
from pliers.filters import (WordStemmingFilter,
                            ColorSpaceFilter)
from pliers.stimuli import ComplexTextStim, ImageStim
from nltk import stem as nls
import pytest
import numpy as np


TEXT_DIR = join(get_test_data_path(), 'text')
IMAGE_DIR = join(get_test_data_path(), 'image')


def test_word_stemming_filter():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)

    # With all defaults (porter stemmer)
    filt = WordStemmingFilter()
    assert isinstance(filt.stemmer, nls.PorterStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    target = ['some', 'sampl', 'text', 'for', 'test', 'annot']
    assert stems == target

    # Try a different stemmer
    filt = WordStemmingFilter(stemmer='snowball', language='english')
    assert isinstance(filt.stemmer, nls.SnowballStemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Handles StemmerI stemmer
    stemmer = nls.WordNetLemmatizer()
    filt = WordStemmingFilter(stemmer=stemmer)
    stemmed = filt.transform(stim)
    stems = [s.text for s in stemmed]
    assert stems == target

    # Fails on invalid values
    with pytest.raises(ValueError):
        filt = WordStemmingFilter(stemmer='nonexistent_stemmer')


def test_color_space_filter():
    stim = ImageStim(join(IMAGE_DIR, 'apple.jpg'))
    filt = ColorSpaceFilter('RGB', 'HSV')
    hsv_stim = filt.transform(stim)
    assert stim.data.shape == hsv_stim.data.shape

    stim = ImageStim(data=[[[0.0, 0.0, 255.0]]])
    # One blue pixel maps to 240 degree hue, 100% sat, 100% value
    hsv_stim = filt.transform(stim)
    assert np.array_equal(hsv_stim.data, ([[[240.0/360.0, 1.0, 255.0]]]))
