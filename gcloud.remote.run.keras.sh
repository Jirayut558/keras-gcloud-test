gcloud ml-engine jobs submit training JOB4A5 \
--module-name=trainer.sentiment_classification_keras \
--package-path=./trainer --job-dir=gs://keras-drive \
--region=asia-east1 --config=./cloudml.yaml --runtime-version=1.10