import base64
import os
from pliers.transformers import Transformer, BatchTransformerMixin
from pliers.utils import (EnvironmentKeyMixin, attempt_to_import,
                          verify_dependencies)


googleapiclient = attempt_to_import('googleapiclient', fromlist=['discovery'])
oauth_client = attempt_to_import('oauth2client.client', 'oauth_client',
                                 ['GoogleCredentials'])


DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'


class GoogleAPITransformer(Transformer, EnvironmentKeyMixin):

    _env_keys = 'GOOGLE_APPLICATION_CREDENTIALS'
    _log_attributes = ('handle_annotations',)

    def __init__(self, discovery_file=None, api_version='v1', max_results=100,
                 num_retries=3, handle_annotations='prefix'):
        verify_dependencies(['googleapiclient', 'oauth_client'])
        if discovery_file is None:
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                raise ValueError("No Google application credentials found. "
                                 "A JSON service account key must be either "
                                 "passed as the discovery_file argument, or "
                                 "set in the GOOGLE_APPLICATION_CREDENTIALS "
                                 "environment variable.")
            discovery_file = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

        self.credentials = oauth_client.GoogleCredentials.from_stream(discovery_file)
        self.max_results = max_results
        self.num_retries = num_retries
        self.service = googleapiclient.discovery.build(self.api_name, api_version,
                                                       credentials=self.credentials,
                                                       discoveryServiceUrl=DISCOVERY_URL)
        self.handle_annotations = handle_annotations
        super(GoogleAPITransformer, self).__init__()

    def _query_api(self, request):
        resource = getattr(self.service, self.resource)()
        request = resource.annotate(body={'requests': request})
        return request.execute(num_retries=self.num_retries)['responses']


class GoogleVisionAPITransformer(BatchTransformerMixin, GoogleAPITransformer):

    api_name = 'vision'
    resource = 'images'
    _batch_size = 10

    def _build_request(self, stims):
        request = []
        for image in stims:
            with image.get_filename() as filename:
                with open(filename, 'rb') as f:
                    img_data = f.read()

            content = base64.b64encode(img_data).decode()
            request.append(
                {
                    'image': {'content': content},
                    'features': [{
                        'type': self.request_type,
                        'maxResults': self.max_results,
                    }]
                })

        return request
