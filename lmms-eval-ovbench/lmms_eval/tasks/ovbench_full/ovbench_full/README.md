---
license: mit
extra_gated_prompt: >-
  You agree to not use the dataset to conduct experiments that cause harm to
  human subjects. Please note that the data in this dataset may be subject to
  other agreements. Before using the data, be sure to read the relevant
  agreements carefully to ensure compliant use. Video copyrights belong to the
  original video creators or platforms and are for academic research use only.
task_categories:
- visual-question-answering
- video-classification
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
modalities:
- Video
- Text
configs:

- config_name: coin
  data_files: json/coin.json
- config_name: hirest
  data_files: json/hirest.json
- config_name: ava
  data_files: json/ava.json
- config_name: tao
  data_files: json/tao.json

language:
- en
size_categories:
- 1K<n<10K
---