import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--task", type=str, help="(train, test)")

    # files
    parser.add_argument("--file", type=str)
    parser.add_argument("--output_file", type=str, default='./output_file')

    # process limits
    parser.add_argument("--limit_papers", type=int, help="Number of papers to process")

    # model
    parser.add_argument("--model_output_path", type=str, default='./model')
    parser.add_argument("--tokenizer_output_path", type=str, default='./tokenizer')

    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()
