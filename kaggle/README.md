# FreshAI Kaggle training

1. Create a Kaggle Notebook and attach a **YOLO detection** dataset. Its
   `data.yaml` must declare every class FreshAI should recognise, such as
   `onion`, `tomato`, `potato`, and `banana`.
2. Turn on **GPU T4 x2** (or another GPU) in Notebook settings.
3. Upload `train_freshai.py` as the notebook code and run it.
4. Download these Notebook Output files:
   - `freshai_yolo.onnx`
   - `freshai_classes.txt`
   - `freshai_best.pt` (keep this for future evaluation/retraining)
5. Replace the Android assets with the downloaded ONNX model and class file,
   then rebuild the application.

The dataset must have correctly labelled bounding boxes. A dataset without an
`onion` class cannot produce an Onion prediction.
