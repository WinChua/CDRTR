init:
	pip install -r requirements.txt

test:
	nosetests -v --nocapture tests --nologcapture
