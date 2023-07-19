.PHONY: docker
docker-cpu:
	docker build -f docker/Dockerfile . -t registry.internal.huggingface.tech/api-inference/community:fast_bert-4
release-cpu:
	docker build -f docker/Dockerfile . -t registry.internal.huggingface.tech/api-inference/community:fast_bert-4 --push
# docker-gpu:
# 	docker build -f docker/Dockerfile.gpu . -t fast_bert --load
# release-gpu:
# 	docker build -f docker/Dockerfile.gpu . -t registry.internal.huggingface.tech/api-inference/community:fast_bert-gpu-1 --push
# run-gpu:
# 	docker run -it  --gpus all --rm -e MODEL_ID=Narsil/finbert -p 8000:80 --mount type=bind,source=/home/nicolas/src/fast_bert/models/,target=/models/ fast_bert fast_bert
