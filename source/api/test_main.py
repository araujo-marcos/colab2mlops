"""
Creator: Ivanovitch Silva
Date: 18 April 2022
API testing
"""
from fastapi.testclient import TestClient
import os
import sys
import pathlib
from source.api.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# a unit test that tests the status code and response 
# for an instance with a low income
def test_get_inference_no_subscribed():

    person = {
        "age": 58,
        "job": 'management',
        "marital": 'married',
        "education": 'tertiary',
        "default": 'no',
        "balance": 2143,
        "housing": 'yes',
        "loan": 'no',
        "contact": 'unknown',
        "day": 5,
        "month": 'may',
        "duration": 261,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": 'unknown'
    }

    r = client.post("/predict", json=person)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "no subscribed"

# a unit test that tests the status code and response 
# for an instance with a high income
def test_get_inference_subscribed():

    person = {
      "age": 60,
      "job": "retired",
      "marital": "married",
      "education": "tertiary",
      "default": "no",
      "balance": 2143,
      "housing": "no",
      "loan": "no",
      "contact": "telephone",
      "day": 5,
      "month": "may",
      "duration": 261,
      "campaign": 1,
      "pdays": -1,
      "previous": 0,
      "poutcome": "unknown"
    }

    r = client.post("/predict", json=person)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "subscribed"