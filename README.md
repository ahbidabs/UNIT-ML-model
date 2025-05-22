run train_rep_model.py to train the rep counting model
run train_model.py to train the exercise detection model
once models are finalized this is the pipeline:
raw sensor data --> 2-second sliding window data sent to exercise detection model --> full exercise set (multiple windows) sen to rep counting model --> return exercise type + rep count
