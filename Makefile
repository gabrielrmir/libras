MODEL_URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

.PHONY: all deps
all: deps
deps: ./models/hand_landmarker.task

./models/hand_landmarker.task:
	mkdir -p ./models
	curl "$(MODEL_URL)" -o ./models/hand_landmarker.task
