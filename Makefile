create_requirements:
	pip freeze --no-color > requirements.txt

black:
	python -m black --line-length 120 *.py

create_dataset:
	python create_dataset.py

train:
	python train.py

test:
	python test.py
