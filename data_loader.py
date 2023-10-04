from datasets import load_dataset


def get_train_data(args):
    train_dataset = load_dataset("scientific_papers", "pubmed", split="train")
    return train_dataset


def get_validation_data(args):
    val_dataset = load_dataset("scientific_papers", "pubmed", split="validation")
    return val_dataset


def get_test_data(args):
    test_dataset = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="test")
    return test_dataset
