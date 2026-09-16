from datasets import load_dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

with open("tinystories_sample.txt", "wb") as output:
    for index, row in enumerate(dataset):
        output.write(row["text"].encode("utf-8"))
        output.write(b"\n\n")

        if index == 499_999:
            break
