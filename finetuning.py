import torch
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, BertForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer, \
    XLMRobertaForQuestionAnswering, AutoModelForQuestionAnswering
import click
from os import listdir
from os.path import join, isdir
import pickle
from utils.utils import compute_metrics
import yaml


def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        stride=128,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    num_missing = 0

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
            num_missing += 1
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    # print('missing: ', num_missing)
    return inputs


def preprocess_validation_examples(examples, tokenizer, padding='max_length'):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        stride=128,
        return_overflowing_tokens=True,
        padding=padding
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


@click.command()
@click.option('--model', default='bert-base-multilingual-cased')
@click.option('--training_languages', default='de')
@click.option('--id', default=-1)
def main(model, training_languages, id):
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    filename = model + '_' + "_".join(sorted(training_languages.split(',')))
    if id == -1:
        num = len([x for x in listdir('.') if isdir(x) and filename in x])
        print(filename)
        print([x for x in listdir('.') if isdir(x) and filename in x])
    else:
        num = id
    out_dir = join(config['model_dir'], filename + '_' + str(num))
    print(out_dir)

    # load xquad data from all languages, merge and then filter them out as testset
    merge_datasets = []
    df_eyetracking_ids = pd.DataFrame()
    for lang in training_languages.split(','):
        df_match = pd.read_pickle(f'utils/mapping_id_qa_{lang}.pkl')
        df_eyetracking_ids = pd.concat([df_eyetracking_ids, df_match])
        merge_datasets.append(load_dataset("xquad", f"xquad.{lang}", split='validation'))

    merged_data = concatenate_datasets(merge_datasets)
    index_test = set(
        [id for eyetracking_id in df_eyetracking_ids.id.map(lambda x: x[:-1]).tolist() for id in merged_data['id'] if
         id.startswith(eyetracking_id)])
    data_train = merged_data.filter(lambda sample: sample['id'] not in index_test)
    data_test = merged_data.filter(lambda sample: sample['id'] in index_test)

    # 90% train, 10% test + validation
    train_valid = data_train.train_test_split(test_size=0.1)
    # gather everyone if you want to have a single DatasetDict
    data = DatasetDict({
        'train': train_valid['train'],
        'validation': train_valid['test'],
        'test': data_test})

    assert (len([id for id in data['train']['id'] if id in data['validation']['id']]) == 0)
    assert (len([id for id in data['train']['id'] if id in data['test']['id']]) == 0)
    assert (len([id for id in data['test']['id'] if id in data['validation']['id']]) == 0)

    data.save_to_disk(out_dir)

    if model == 'bert-base-multilingual-cased':
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = BertForQuestionAnswering.from_pretrained(model)
    elif model == 'xlm-roberta-base' or model == 'xlm-roberta-large':
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = XLMRobertaForQuestionAnswering.from_pretrained(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForQuestionAnswering.from_pretrained(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)

    tokenized_train = data['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=data['train'].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    tokenized_val = data["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=data["validation"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(out_dir)

    predictions, _, _ = trainer.predict(tokenized_val)
    start_logits, end_logits = predictions
    results_val = compute_metrics(start_logits, end_logits, tokenized_val, data["validation"])
    print(f'val_set ({len(data["validation"])}):', results_val)
    pickle.dump(results_val, open(join(out_dir, "val.p"), "wb"))

    tokenized_test = data["test"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=data["test"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    predictions, _, _ = trainer.predict(tokenized_test)
    start_logits, end_logits = predictions
    results_test = compute_metrics(start_logits, end_logits, tokenized_test, data["test"])
    print(f'test_set ({len(data["test"])}):', results_test)
    pickle.dump(results_test, open(join(out_dir, "test.p"), "wb"))


if __name__ == '__main__':
    main()
