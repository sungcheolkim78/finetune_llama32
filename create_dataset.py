import pandas as pd
import random


def create_random_math_questions(max_int: int = 1_000):
    num1 = random.randint(1, max_int)
    num2 = random.randint(1, max_int)
    res = num1 + num2

    return num1, num2, res


def main(n: int = 10_000, filename: str = "math_dataset.pkl", train: bool = True):
    dataset = []
    for _ in range(n):
        num1, num2, res = create_random_math_questions()

        prompt = f"What is the answer to the following math question: {num1} + {num2}?"
        dataset.append((prompt, str(res)))

    # Check the dataset
    # df = pd.DataFrame(dataset, columns=["prompt", "target"])

    new_dataset = []
    for prompt, response in dataset:
        new_row = [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}]
        new_dataset.append(new_row)

    if train:
        df = pd.DataFrame({"conversations": new_dataset})
    else:
        df = pd.DataFrame(new_dataset, columns=["prompt", "gt"])

    # Save as pickle
    df.to_pickle(filename)


if __name__ == "__main__":
    # For training
    main(n=10_000, filename="math_dataset_train.pkl", train=True)

    # For testing
    main(n=1_000, filename="math_dataset_test.pkl", train=False)

