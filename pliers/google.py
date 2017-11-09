import os
from pliers.transformers import Transformer, BatchTransformerMixin
from pliers.utils import (EnvironmentKeyMixin, attempt_to_import,
                          verify_dependencies)


google_cloud = attempt_to_import('google.cloud', 'google_cloud', ['vision'])


class GoogleAPITransformer(Transformer, EnvironmentKeyMixin):

    _env_keys = 'GOOGLE_APPLICATION_CREDENTIALS'
    _log_attributes = ('handle_annotations',)

    def __init__(self, discovery_file=None, max_results=100,
                 handle_annotations='prefix'):
        verify_dependencies(['google_cloud'])
        if discovery_file is None:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise ValueError("No Google application credentials found. "
                                 "A JSON service account key must be either "
                                 "passed as the discovery_file argument, or "
                                 "set in the GOOGLE_APPLICATION_CREDENTIALS "
                                 "environment variable.")
            discovery_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

        self.discovery_file = discovery_file
        self.max_results = max_results
        self.handle_annotations = handle_annotations
        super(GoogleAPITransformer, self).__init__()


class GoogleVisionAPITransformer(BatchTransformerMixin, GoogleAPITransformer):

    _batch_size = 10

    def __init__(self, **kwargs):
        super(GoogleVisionAPITransformer, self).__init__(**kwargs)
        creds = google_cloud.vision.Client.from_service_account_json(self.discovery_file)._credentials
        self.client = google_cloud.vision.ImageAnnotatorClient(credentials=creds)

    def _build_request(self, stims):
        request = []
        for image in stims:
            with image.get_filename() as filename:
                with open(filename, 'rb') as f:
                    img_data = f.read()

            content = img_data
            request.append(
                {
                    'image': {'content': content},
                    'features': [{
                        'type': self.request_type,
                        'max_results': self.max_results,
                    }]
                })

        return request
