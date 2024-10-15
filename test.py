"""
Run inference on a test sample for the new fine-tuned model
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, TextStreamer
from peft import PeftModel
import pandas as pd

SAVED_MODEL_FOLDER = "model_2024-10-15_09-49-14"
SAVED_ADAPTER_FOLDER = "output/checkpoint-24"


def main(filename: str = "math_dataset_test.pkl"):
    df_test = pd.read_pickle(filename)

    model, tokenizer = load_model(with_lora=True)
    responses = generate_responses(model, tokenizer, df_test)

    df_test["response_finetuned_model"] = responses

    model, tokenizer = load_model(with_lora=False)
    responses = generate_responses(model, tokenizer, df_test)

    df_test["response_non_finetuned_model"] = responses

    now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    df_test.to_pickle(filename.replace('.pkl', f'_final_{now}.pkl'))
    print(df_test.head())


def load_model(with_lora: bool = True):
    max_seq_length = 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SAVED_MODEL_FOLDER,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.for_inference(model)

    if with_lora:
        model = PeftModel.from_pretrained(model, SAVED_ADAPTER_FOLDER)

    return model, tokenizer


def generate_responses(model, tokenizer, df):
    messages = df["prompt"].tolist()

    responses = []
    for message in messages:
        # need to wrap in a list
        message = [message]
        inputs = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer)
        response = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=64, use_cache=True)

        response_txt = tokenizer.decode(response[0], skip_special_tokens=True)
        responses.append(response_txt)

    return responses


if __name__ == "__main__":
    main("math_dataset_test.pkl")
