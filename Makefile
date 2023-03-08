.PHONY: docker
docker:
	docker build -f docker/Dockerfile . -t fast_bert
