from time import perf_counter
import os

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

HF_MODEL_NAME = "ProsusAI/finbert"

if __name__ == "__main__":
    # tf.config.experimental.enable_mlir_graph_optimization = True
    # tf.config.optimizer.set_experimental_options(
    #     {
    #         "debug_stripper": True,
    #         "constant_folding": True,
    #         "shape_optimization": True,  # Target hardware dependant
    #         "arithmetic_optimization": True,  # Target hardware dependant
    #         "dependency_optimization": True,
    #         "loop_optimization": True,
    #         "function_optimization": True,
    #         "remapping": True,  # Target hardware dependant
    #     }
    # )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, padding_side="left")
    # model = TFAutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

    tokenization_kwargs = {"return_tensors": "tf"}

    default_string = "test eqwlewqk ewqlke qwlkeqwl ewlqke qwlke eklqwekwqlek qwlkeqwl ekqwlk eqwlke qwlke qwlke qwlkelqw elqwkelwk elkw elkqwel qwel qwle kqwejqwkehjqwjkeh qwjkhe qwjkhekqweh qwjkeh qwjkeh qwkje"

    string = os.getenv("STRING", default_string)
    tokenized_inputs = tokenizer(string, **tokenization_kwargs)

    # tf.keras.models.save_model(
    #     model,
    #     "fixtures/artifacts/bert",
    #     signatures={"classify": model.serving.get_concrete_function()},
    #     include_optimizer=False,
    # )

    model_ = tf.keras.models.load_model("fixtures/artifacts/bert")
    classify_f = model_.signatures["classify"]
    _ = classify_f(**tokenized_inputs)

    for _ in range(10):
        start = perf_counter()
        _ = classify_f(**tokenized_inputs)
        end = perf_counter()

        print(f"Inference took {round((end - start) * 1000, 2)} ms")
