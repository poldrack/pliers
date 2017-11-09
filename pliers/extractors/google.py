''' Google API-based feature extraction classes. '''

from pliers.extractors.image import ImageExtractor
from pliers.google import GoogleVisionAPITransformer
from pliers.extractors.base import ExtractorResult
import numpy as np


class GoogleVisionAPIExtractor(GoogleVisionAPITransformer, ImageExtractor):

    ''' Base class for all Extractors that use the Google Vision API. '''

    VERSION = '1.0'

    def _extract(self, stims):
        request = self._build_request(stims)
        responses = self.client.batch_annotate_images(request).responses

        results = []
        for i, response in enumerate(responses):
            if response and hasattr(response, self.response_object):
                annotations = getattr(response, self.response_object)
                features, values = self._parse_annotations(annotations)
                values = [values]
                results.append(ExtractorResult(values, stims[i], self,
                                               features=features))
            elif 'error' in response:
                raise Exception(response['error']['message'])

            else:
                results.append(ExtractorResult([[]], stims[i], self,
                                               features=[]))

        return results


class GoogleVisionAPIFaceExtractor(GoogleVisionAPIExtractor):

    ''' Identifies faces in images using the Google Cloud Vision API. '''

    request_type = 'FACE_DETECTION'
    response_object = 'face_annotations'

    def _parse_annotations(self, annotations):
        features = []
        values = []

        if self.handle_annotations == 'first':
            annotations = [annotations[0]] if annotations else []

        for i, annotation in enumerate(annotations):
            data_dict = {}
            for field, val in annotation.ListFields():
                field = field.name
                if 'confidence' in field:
                    data_dict['face_' + field] = val
                elif 'ounding_poly' in field:
                    for j, vertex in enumerate(val.vertices):
                        name = '%s_vertex%d_%s' % (field, j+1, 'x')
                        val = vertex.x if vertex.x else np.nan
                        data_dict[name] = val
                        name = '%s_vertex%d_%s' % (field, j+1, 'y')
                        val = vertex.y if vertex.y else np.nan
                        data_dict[name] = val
                elif field == 'landmarks':
                    for lm in val:
                        name = 'landmark_' + repr(lm.type) + '_%s'
                        lm_pos = {name %
                                  k.name: v for (k, v) in lm.position.ListFields()}
                        data_dict.update(lm_pos)
                elif 'likelihood' in field:
                    data_dict[field] = (val - 1) / 4.0
                else:
                    data_dict[field] = val

            names = list(data_dict.keys())
            if self.handle_annotations == 'prefix' and len(annotations) > 1:
                names = ['face%d_%s' % (i+1, n) for n in names]
            features += names
            values += list(data_dict.values())

        return features, values


class GoogleVisionAPILabelExtractor(GoogleVisionAPIExtractor):

    ''' Labels objects in images using the Google Cloud Vision API. '''

    request_type = 'LABEL_DETECTION'
    response_object = 'label_annotations'

    def _parse_annotations(self, annotations):
        features = []
        values = []
        for annotation in annotations:
            features.append(annotation.description)
            values.append(annotation.score)
        return features, values


class GoogleVisionAPIPropertyExtractor(GoogleVisionAPIExtractor):

    ''' Extracts image properties using the Google Cloud Vision API. '''

    request_type = 'IMAGE_PROPERTIES'
    response_object = 'image_properties_annotation'

    def _parse_annotations(self, annotation):
        colors = annotation.dominant_colors.colors
        features = []
        values = []
        for color in colors:
            rgb = color.color
            features.append((rgb.red, rgb.green, rgb.blue))
            values.append(color.score)
        return features, values


class GoogleVisionAPISafeSearchExtractor(GoogleVisionAPIExtractor):

    ''' Extracts safe search detection using the Google Cloud Vision API. '''

    request_type = 'SAFE_SEARCH_DETECTION'
    response_object = 'safe_search_annotation'

    def _parse_annotations(self, annotation):
        keys = []
        vals = []
        print annotation
        for k, v in annotation.ListFields():
            keys.append(k.name)
            vals.append((v - 1) / 4.0)
        return keys, vals


class GoogleVisionAPIWebEntitiesExtractor(GoogleVisionAPIExtractor):

    ''' Extracts web entities using the Google Cloud Vision API. '''

    request_type = 'WEB_DETECTION'
    response_object = 'web_detection'

    def _parse_annotations(self, annotations):
        features = []
        values = []
        if hasattr(annotations, 'web_entities'):
            for annotation in annotations.web_entities:
                if hasattr(annotation, 'description') and hasattr(annotation, 'score'):
                    features.append(annotation.description)
                    values.append(annotation.score)
        return features, values
