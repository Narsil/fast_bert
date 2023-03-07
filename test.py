import datetime

start = datetime.datetime.now()
import torch

print(f"Loaded torch {datetime.datetime.now() - start}")

from transformers import pipeline

print(f"Loaded transformers {datetime.datetime.now() - start}")


pipe = pipeline(task="text-classification", model="Narsil/finbert")
print(f"Loaded in {datetime.datetime.now() - start}")
inf_start = datetime.datetime.now()
out = pipe("My name is")
print(f"Inference took: {(datetime.datetime.now() - inf_start)}")
print(out)
print(f"Ran in {(datetime.datetime.now() - start)}")
