import datetime
import os

start = datetime.datetime.now()
import torch

print(f"Loaded torch {datetime.datetime.now() - start}")

from transformers import pipeline

print(f"Loaded transformers {datetime.datetime.now() - start}")


pipe = pipeline(task="text-classification", model="Narsil/finbert")
default_string = "test eqwlewqk ewqlke qwlkeqwl ewlqke qwlke eklqwekwqlek qwlkeqwl ekqwlk eqwlke qwlke qwlke qwlkelqw elqwkelwk elkw elkqwel qwel qwle kqwejqwkehjqwjkeh qwjkhe qwjkhekqweh qwjkeh qwjkeh qwkje"
string = os.getenv("STRING", default_string)
print(f"Loaded in {datetime.datetime.now() - start}")
for i in range(5):
    inf_start = datetime.datetime.now()
    out = pipe(string)
    print(f"Inference took: {(datetime.datetime.now() - inf_start)}")
print(out)
print(f"Ran in {(datetime.datetime.now() - start)}")
