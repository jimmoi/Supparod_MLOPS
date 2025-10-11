python mlops_pipeline/scripts/01_data_validation.py
python mlops_pipeline/scripts/02_preprocess_and_training.py local 32 0.001 2
python mlops_pipeline/scripts/03_transition_model.py local
python mlops_pipeline/scripts/04_load_and_predict.py