from pliers.filters import FrameSamplingFilter
from pliers.extractors import (GoogleVisionAPIFaceExtractor,
                               GoogleVisionAPILabelExtractor,
                               GoogleVisionAPIPropertyExtractor,
                               GoogleVisionAPISafeSearchExtractor,
                               GoogleVisionAPIWebEntitiesExtractor,
                               ExtractorResult)
from pliers.extractors.google import GoogleVisionAPIExtractor
from pliers.stimuli import ImageStim, VideoStim
import pytest
import json
from os.path import join
from ..utils import get_test_data_path
import numpy as np


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_extractor_inits():
    ext = GoogleVisionAPIExtractor()
    assert ext.max_results == 100
    assert ext.client is not None


# @pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
# def test_google_vision_api_face_extractor_inits():
#     ext = GoogleVisionAPIFaceExtractor()
#     assert ext.max_results == 100
#     assert ext.client is not None

#     # Test parsing of individual response
#     filename = join(
#         get_test_data_path(), 'payloads', 'google_vision_api_face_payload.json')
#     response = json.load(open(filename, 'r'))
#     features, data = ext._parse_annotations(response['face_annotations'])
#     assert len(features) == len(data)
#     assert data[features.index('anger_likelihood')] == 0.0
#     assert data[
#         features.index('landmark_LEFT_EYE_BOTTOM_BOUNDARY_y')] == 257.023
#     assert np.isnan(data[features.index('bounding_poly_vertex2_y')])


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_face_extractor():
    ext = GoogleVisionAPIFaceExtractor()
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'joy_likelihood' in result.columns
    assert result['joy_likelihood'][0] == 1.0
    assert result['face_detection_confidence'][0] > 0.7


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_multiple_face_extraction():
    filename = join(get_test_data_path(), 'image', 'thai_people.jpg')
    stim = ImageStim(filename)
    # Only first record
    ext = GoogleVisionAPIFaceExtractor(handle_annotations='first')
    result1 = ext.transform(stim).to_df()
    assert 'joy_likelihood' in result1.columns
    # All records
    ext = GoogleVisionAPIFaceExtractor(handle_annotations='prefix')
    result2 = ext.transform(stim).to_df()
    assert 'face2_joy_likelihood' in result2.columns
    assert result2.shape[1] > result1.shape[1]


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_face_batch():
    obama_file = join(get_test_data_path(), 'image', 'obama.jpg')
    people_file = join(get_test_data_path(), 'image', 'thai_people.jpg')
    stims = [ImageStim(obama_file), ImageStim(people_file)]
    ext = GoogleVisionAPIFaceExtractor(handle_annotations='first')
    result = ext.transform(stims)
    result = ExtractorResult.merge_stims(result)
    assert 'joy_likelihood' in result.columns
    assert result['joy_likelihood'][0] == 1.0
    assert result['joy_likelihood'][1] == 1.0

    video = VideoStim(join(get_test_data_path(), 'video', 'obama_speech.mp4'))
    conv = FrameSamplingFilter(every=10)
    video = conv.transform(video)
    result = ext.transform(video)
    result = ExtractorResult.merge_stims(result)
    assert 'joy_likelihood' in result.columns
    assert result.shape == (11, 137)

    video = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
    video = conv.transform(video)
    result = ext.transform(video)
    result = ExtractorResult.merge_stims(result)
    assert 'joy_likelihood' not in result.columns
    assert result.shape == (17, 7)


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_label_extractor():
    ext = GoogleVisionAPILabelExtractor()
    filename = join(get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'apple' in result.columns
    assert result['apple'][0] > 0.75


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_properties_extractor():
    ext = GoogleVisionAPIPropertyExtractor()
    filename = join(get_test_data_path(), 'image', 'apple.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert (158, 13, 29) in result.columns
    assert np.isfinite(result[(158, 13, 29)][0])


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_safe_search():
    ext = GoogleVisionAPISafeSearchExtractor()
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'adult' in result.columns
    assert result['violence'][0] == 0.0


@pytest.mark.skipif("'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ")
def test_google_vision_api_web_entities():
    ext = GoogleVisionAPIWebEntitiesExtractor()
    filename = join(get_test_data_path(), 'image', 'obama.jpg')
    stim = ImageStim(filename)
    result = ext.transform(stim).to_df()
    assert 'Barack Obama' in result.columns
