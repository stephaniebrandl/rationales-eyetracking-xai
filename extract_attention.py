import pandas as pd
from transformers import AutoTokenizer, BertForQuestionAnswering, XLMRobertaForQuestionAnswering, Trainer, \
    AutoModelForQuestionAnswering
from datasets import load_from_disk
import torch
import numpy as np
from os.path import join
from finetuning import preprocess_validation_examples, compute_metrics
import scipy
from tqdm import tqdm
from utils.attention_graph_util import compute_joint_attention
import click


# For the attention baseline, we fixed several experimental choices (see below) which might affect the results.
def calculate_relative_attention(tokens, attention, sep_token,
                                 special_tokens=['[PAD]', '[CLS]', '[SEP]', '</s>', '<s>', '<pad>']):
    # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
    # I also tried the sum once, but the result was even worse
    mean_attention = np.mean(attention, axis=0)

    # remove question
    sep_index = tokens.index(sep_token)
    tokens = tokens[sep_index + 1:]
    mean_attention = mean_attention[sep_index + 1:, sep_index + 1:]
    # mean_attention = mean_attention[sep_index + 1:]

    tokens_out = []
    index_del = []

    for (itok, tok) in enumerate(tokens):
        if tok in special_tokens:
            index_del.append(itok)
        else:
            tokens_out.append(tok)

    mean_attention = np.delete(mean_attention, index_del, 0)
    # 2. For each word, we sum over the attention to the other words to determine relative importance
    sum_attention = np.sum(mean_attention, axis=0)

    sum_attention = np.delete(sum_attention, index_del, 0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    relative_attention = scipy.special.softmax(sum_attention)

    return tokens_out, relative_attention


def compute_rollout(all_attentions, layer):
    _attentions = [att.detach().cpu().numpy() for att in all_attentions]
    attentions_mat = np.asarray(_attentions)[:, 0]

    res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
    res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None, ...]
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]

    joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)

    layer_num = -1 if layer == 'last' else layer

    return joint_attentions[layer_num]


def evaluate_model(tokenized_test, test_data, model, model_folder):
    trainer = Trainer(model)
    predictions, _, _ = trainer.predict(tokenized_test)
    start_logits, end_logits = predictions

    df = pd.DataFrame(columns=['id', 'exact', 'f1'])
    for ii, batch in enumerate(range(len(tokenized_test))):
        results = compute_metrics(start_logits[batch][None, :], end_logits[batch][None, :],
                                  tokenized_test.select([batch]),
                                  test_data.select([batch]))
        df.loc[ii] = [tokenized_test.select([batch])['example_id'][0], results['exact_match'], results['f1']]
    df.to_pickle(join(model_folder, 'eval_test.pkl'))

    return df


@click.command()
@click.option('--modelname', default='bert-base-multilingual-cased')
@click.option('--lang', default='en')
@click.option('--id', default=0)
def main(modelname, lang, id):
    COMPUTE_ROLLOUT = True

    model_folder = f'../models/{modelname}_{lang}_{id}'
    print(model_folder)

    if modelname == 'distilbert-base-multilingual-cased':
        model = AutoModelForQuestionAnswering.from_pretrained(model_folder)
        sep_token = '[SEP]'
        layers = [0, 5, 'last']
    elif modelname == 'bert-base-multilingual-cased':
        model = BertForQuestionAnswering.from_pretrained(model_folder)
        sep_token = '[SEP]'
        layers = [0, 5, 'last']
    elif modelname == 'xlm-roberta-base':
        model = XLMRobertaForQuestionAnswering.from_pretrained(model_folder)
        sep_token = '</s>'
        layers = [0, 5, 'last']
    elif modelname == 'xlm-roberta-large':
        model = XLMRobertaForQuestionAnswering.from_pretrained(model_folder)
        sep_token = '</s>'
        layers = [0, 11, 'last']
    else:
        raise NotImplementedError()

    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    dataset = load_from_disk(join(model_folder, 'test'))

    tokenized_test = dataset.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    try:
        df_eval = pd.read_pickle(join(model_folder, 'eval_test.pkl'))
    except FileNotFoundError:
        df_eval = evaluate_model(tokenized_test, dataset, model, model_folder)

    df_attention = pd.DataFrame(columns=['id', 'tokens', 'model'])

    for layer in layers:
        df_attention[f'attention_{layer}'] = None
        df_attention[f'rollout_{layer}'] = None
        layer_num = -1 if layer == 'last' else layer
        for ii in tqdm(range(len(tokenized_test))):
            output = model(input_ids=torch.tensor(tokenized_test['input_ids'])[ii, None].to(device),
                           output_attentions=True)
            attention = output['attentions']
            input_id_list = torch.tensor(tokenized_test['input_ids'])[ii].numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_id_list)
            if COMPUTE_ROLLOUT:
                rollout = compute_rollout(attention, layer)
                _, relative_rollout = calculate_relative_attention(tokens,
                                                                   rollout[None, :, :],
                                                                   sep_token)
            assert (len(tokens) == len(attention[0][0][0]))
            tokens, relative_attention = calculate_relative_attention(tokens,
                                                                      attention[layer_num][0].detach().cpu().numpy(),
                                                                      sep_token)
            df_attention.loc[ii, 'id'] = tokenized_test['example_id'][ii]
            df_attention.at[ii, 'tokens'] = tokens
            df_attention.at[ii, f'attention_{layer}'] = relative_attention
            df_attention.at[ii, f'rollout_{layer}'] = relative_rollout
            df_attention.at[ii, 'model'] = df_eval[df_eval.id == tokenized_test['example_id'][ii]]['f1'].values[0]
    df_attention.to_pickle(join(model_folder, "attention.pkl"))


if __name__ == '__main__':
    main()
