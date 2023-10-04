import torch

from datasets import load_dataset, load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration

# load pubmed
from data_loader import get_test_data


def generate_answer(batch):
    global tokenizer
    global model

    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt",
                            truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask,
                                            global_attention_mask=global_attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batch


def process(args):
    global tokenizer
    global model

    test_dataset = get_test_data(args)

    # load tokenizer
    tokenizer = LEDTokenizer.from_pretrained(args.tokenizer_output_path)
    model = LEDForConditionalGeneration.from_pretrained(args.model_output_path).to("cuda").half()

    result = test_dataset.map(generate_answer, batched=True, batch_size=args.batch_size)

    # load rouge
    rouge = load_metric("rouge")

    result.save_to_disk(args.output_file)
    print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"],
                                   rouge_types=["rouge2"])["rouge2"].mid)
