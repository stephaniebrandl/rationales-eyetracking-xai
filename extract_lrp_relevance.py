import pandas as pd
from transformers import AutoTokenizer
from datasets import load_from_disk
import torch
import numpy as np
from os.path import join
from finetuning import preprocess_validation_examples
from tqdm import tqdm
from utils.transformer_lrp import BertForQuestionAnsweringExplainer, DistilbertForQuestionAnsweringExplainer, Config, \
    ConfigGradienxInput
from utils.lrp_utils import get_best_answer, plot_conservation
import click
from utils.utils import get_model


@click.command()
@click.option('--model', default='bert-base-multilingual-cased')
@click.option('--lang', default='en')
@click.option('--id', default=0)
@click.option('--case', default='lrp')
# python extract_lrp_relevance.py --model xlm-roberta-large --lang en
# python extract_lrp_relevance.py --model xlm-roberta-base --lang en
# python extract_lrp_relevance.py --model distilbert-base-multilingual-cased --lang en

def main(model, lang, id, case):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model_folder = '../models/' + model + '_' + lang + '_' + str(id)

    model, base = get_model(model_folder)

    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    pretrained_embeds = getattr(getattr(model, base), 'embeddings')

    # Load explainable model
    if case == 'lrp':
        config = Config(model_folder, device)
    elif case == 'gi':
        config = ConfigGradienxInput(model_folder, device)

    if 'distilbert' in model_folder:
        model_xai = DistilbertForQuestionAnsweringExplainer(config, pretrained_embeds)
    else:
        model_xai = BertForQuestionAnsweringExplainer(config, pretrained_embeds)
    state_dict_src = model.state_dict()
    renamed_state_dict = model_xai.match_state_dicts(state_dict_src)

    model_xai.load_state_dict(renamed_state_dict)
    model_xai.eval()
    model_xai.to(device)

    inputs_ids_test = torch.tensor(np.array([1, 2, 3, 4, 100])).to(device).unsqueeze(0)

    dataset = load_from_disk(join(model_folder, 'test'))

    tokenized_test = dataset.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    df_attention = pd.DataFrame(columns=['id', 'tokens', 'attention'])

    contexts = [{"id": ex["id"], "context": ex["context"]} for ex in dataset]
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]

    Rs = []
    Ls = []

    for ii in tqdm(range(len(tokenized_test))):
        # Input sentence without padding
        inputs = torch.tensor(tokenized_test['input_ids'])[ii, None]
        attention_mask = torch.tensor(tokenized_test['attention_mask'])[ii, None]
        inputs_ = inputs[attention_mask == 1].unsqueeze(0)
        tokens = tokenizer.convert_ids_to_tokens(inputs.squeeze())

        output = model(input_ids=inputs_.to(device), output_attentions=False)

        example_id = tokenized_test['example_id'][ii]
        context = [k for k in contexts if k['id'] == example_id][0]['context']

        predicted_answer = get_best_answer(start_logit=output['start_logits'].detach().squeeze().cpu().numpy(),
                                           end_logit=output['end_logits'].detach().squeeze().cpu().numpy(),
                                           offsets=tokenized_test['offset_mapping'][ii],
                                           context=context,
                                           example_id=example_id)

        logit_mask = predicted_answer["mask"]

        # Extract relevance (based on model prediction)
        outs = model_xai.forward_and_explain(inputs_.cuda(), cl=logit_mask)
        relevance = outs['R'].squeeze().detach().cpu().numpy()

        Rs.append(relevance.sum())
        Ls.append(model_xai.logit.detach().cpu().numpy().sum())

        df_attention.loc[ii] = [tokenized_test['example_id'][ii], tokens, relevance]

    if case == 'lrp':
        df_attention.to_pickle(join(model_folder, "relevance.pkl"))
        plot_conservation(Ls, Rs, join(model_folder, "conservation_lrp.png"))
    elif case == 'gi':
        df_attention.to_pickle(join(model_folder, "gradientsinput.pkl"))
        plot_conservation(Ls, Rs, join(model_folder, "conservation_gi.png"))


if __name__ == '__main__':
    main()
