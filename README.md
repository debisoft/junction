---
title: Junction
emoji: 🏃
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 3.40.1
app_file: app.py
pinned: false
license: lgpl
---

# junction
BiteBuddies AI/ML Stack🤖

Tinder-style Recommender System/Collaborative Filtering

Testing:

        curl -X POST https://debisoft-junction.hf.space/api/predict -H 'Content-Type: application/json' -d '{"data": [<name>,<body_profile_type>]}

body_profile_type => [0-4]

        Eg. curl -X POST https://debisoft-junction.hf.space/api/predict -H 'Content-Type: application/json' -d '{"data": ["David",4]}

Checkout the API Demo at
https://huggingface.co/spaces/debisoft/junction
