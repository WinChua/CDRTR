export PYTHONPATH=..

generatevoca:
	python -m CDRTR.preprocess generatevoca --mode $(MODE) --fields asin,reviewerID,overall,reviewText,unixReviewTime --data_dir $(DATA)

extractinfo:
	python -m CDRTR.preprocess extractinfo --mode $(MODE)

preprocess: generatevoca extractinfo

sentitrain:
	python -m CDRTR.core.DeepModel.SentiRec --dir data/preprocess --domain Musi --filter_size 3,5,7,11 --epoches 10
