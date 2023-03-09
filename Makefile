.PHONY: docker
docker:
	docker build -f docker/Dockerfile . -t fast_bert
release:
	docker build -f docker/Dockerfile . -t registry.internal.huggingface.tech/api-inference/community:fast_bert --push
