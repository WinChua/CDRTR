export PYTHONPATH=..

generatevoca:
	python -m CDRTR.preprocess generatevoca --mode $(MODE) --fields asin,reviewerID,overall,reviewText

extractinfo:
	python -m CDRTR.preprocess extractinfo --mode $(MODE)

preprocess: generatevoca extractinfo
