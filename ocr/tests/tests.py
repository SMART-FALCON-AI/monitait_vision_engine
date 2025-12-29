import unittest
import requests
import os

HOST = os.environ.get('HOST', 'http://0.0.0.0:3001')


class TestAPI(unittest.TestCase):
    def test_health(self):
        r = requests.get(HOST + '/api/v1/health')
        self.assertEqual(r.status_code, 200)

    def test_set_languages(self):
        r = requests.post(HOST + '/api/v1/set-languages', data={'languages': ['en', 'fr']})
        self.assertEqual(r.status_code, 200)

    def test_get_languages(self):
        r = requests.get(HOST + '/api/v1/get-languages')
        self.assertEqual(r.status_code, 200)

    def test_predict(self):
        r = requests.post(HOST + '/api/v1/predict', files={'image': open('sample.jpg', 'rb')})
        self.assertEqual(r.status_code, 200)



