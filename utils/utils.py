import itertools
import numpy as np
from utils.lrp_utils import get_best_answer

import torch
import pandas as pd
import collections
import evaluate
from tqdm.auto import tqdm
import gc
import yaml

metric = evaluate.load("squad")

from transformers import AutoTokenizer, BertForQuestionAnswering, XLMRobertaForQuestionAnswering, \
    AutoModelForQuestionAnswering

import re
from scipy.ndimage.filters import gaussian_filter1d

from os import listdir, makedirs
from os.path import isdir, join


def get_plotdict():
    pandas_dict = {'attention_0': '1st-layer att.',
                   'attention_last': 'last-layer att.',
                   'rollout_last': 'last-layer roll.',
                   'relevance': 'LRP',
                   'gradientsinput': 'Grad.xInput',
                   'eyetracking_cont': 'Gaze',
                   'eyetracking': 'Gaze',
                   'mbert': 'mBERT',
                   'distil': 'distil-mBERT',
                   'xlmr': 'XLMR',
                   'xlmr_large': 'XLMR-L',
                   'bert': 'BERT',
                   'distil_mono': 'distil-BERT'
                   }
    labels_dict = {'len_answer': 'Length of answer',
                   'position_answer': 'Relative position of answer',
                   'len_text': 'Length of text'}
    xlabels_dict = {'len_answer': "",
                    'position_answer': 'Bins in %',
                    'len_text': ''
                    }
    ticks_dict = {'position_answer': ['0-19', '20-50', '51-70', '71-100']}

    return pandas_dict, labels_dict, xlabels_dict, ticks_dict


def get_resultsdir(lang, correct_answers, subset, filter_quality, threshold, workers, filter_vision, vision):
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    resultsdir = join(config['results_dir'], f"evaluation_{lang}")
    resultsdir = resultsdir + f'_{float(threshold)}' if filter_quality and threshold is not None else resultsdir
    resultsdir = resultsdir + '_correct-answers' if correct_answers else resultsdir
    resultsdir = resultsdir + '_subset' if subset else resultsdir
    resultsdir = resultsdir + '_lab' if workers == 'lab' else resultsdir
    resultsdir = resultsdir + '_volunteer' if workers == 'volunteer' else resultsdir
    resultsdir = resultsdir + f'_vision_{vision}' if filter_vision else resultsdir
    return resultsdir


def setup_dir(path):
    if not isdir(path):
        makedirs(path)


def load_rationales(file, modelbase, correct_answers, F1):
    df_rationales = pd.read_pickle(f"./rationales/{file}")
    df_rationales = df_rationales.rename(columns={
        'attention_11': 'attention_middle', 'attention_5': 'attention_middle',
        'rollout_11': 'rollout_middle', 'rollout_5': 'rollout_middle'})

    if correct_answers and modelbase != 'average':
        df_rationales = df_rationales.query(f'model_f1 >= {F1}')
        # df_rationales = df_rationales.query('model_f1 >= 0')

    return df_rationales


def get_files(modelname, correct_answers, filter_quality, threshold, workers,
              filter_vision, vision):
    suffix = ''
    suffix = suffix + '_lab' if workers == 'lab' else suffix
    suffix = suffix + '_volunteer' if workers == 'volunteer' else suffix
    suffix = suffix + '_correct-answers' if correct_answers else suffix
    suffix = suffix + f'_{float(threshold)}' if filter_quality and threshold is not None else suffix
    suffix = suffix + f'_vision_{vision}' if filter_vision else suffix

    if suffix == '':
        files = [file for file in listdir('../rationales') if
                 file.startswith(modelname)
                 and not file.endswith("_correct-answers.pkl")
                 and not file.endswith("normal.pkl")
                 and not file.endswith("glasses.pkl")
                 and "volunteer" not in file
                 and "lab" not in file
                 and not file.split(".")[0].split("_")[-1].isdigit()]

    else:
        files = [file for file in listdir('../rationales') if file.startswith(modelname)
                 and file.endswith(suffix + ".pkl")]
        # and "volunteer" not in file
        # and "lab" not in file]

    return files


def get_modelname(base, lang, imodel):
    model_dict = {
        'mbert': f'bert-base-multilingual-cased_{lang}_{imodel}',
        'bert': f'bert-base-cased_{lang}_{imodel}',
        'roberta': f'roberta-base_{lang}_{imodel}',
        'xlmr': f'xlm-roberta-base_{lang}_{imodel}',
        'xlmr_large': f'xlm-roberta-large_{lang}_{imodel}',
        'distil': f'distilbert-base-multilingual-cased_{lang}_{imodel}',
        'distil_mono': f'distilbert-base-uncased_{lang}_{imodel}',
        'average': f'average_{lang}'
    }

    return model_dict[base]


def get_model(model_folder):
    if 'distilbert-base-uncased' in model_folder:
        model = AutoModelForQuestionAnswering.from_pretrained(model_folder)
        base = 'distilbert_mono'
    elif 'distilbert-base-multilingual' in model_folder:
        model = AutoModelForQuestionAnswering.from_pretrained(model_folder)
        base = 'distilbert'
    elif 'bert-base-multilingual' in model_folder:
        model = BertForQuestionAnswering.from_pretrained(model_folder)
        base = 'mbert'
    elif 'bert-base-cased' in model_folder:
        model = BertForQuestionAnswering.from_pretrained(model_folder)
        base = 'bert'
    elif 'xlm-roberta-large' in model_folder:
        model = XLMRobertaForQuestionAnswering.from_pretrained(model_folder)
        base = 'xlmr-large'
    elif 'xlm-roberta-base' in model_folder:
        model = XLMRobertaForQuestionAnswering.from_pretrained(model_folder)
        base = 'xlmr'
    elif 'roberta-base_' in model_folder:
        model = XLMRobertaForQuestionAnswering.from_pretrained(model_folder)
        base = 'roberta'
    return model, base


def get_tokenized_answer_mask(answer, input_ids, tokenizer):
    answer_ids = tokenizer(answer)['input_ids']
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)[1:-1]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    string = '_'.join(tokens)
    pattern = '_'.join(answer_tokens)
    span = re.search(pattern, string, flags=0)

    if span is None:
        return None
    else:
        start_idx = len(string[:span.start() - 1].split('_'))

        end_idx = len(string[:span.end()].split('_'))

        mask = np.zeros_like(input_ids)
        mask[start_idx:end_idx] = 2
        mask[start_idx] = 1
        mask[end_idx - 1] = 3

    return mask


def preprocess_validation_examples(examples):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        padding="max_length",
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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)

    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):

        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = np.array(features[feature_index]["offset_mapping"]).squeeze().tolist()

            start_indexes = np.argsort(start_logit)[-1:-n_best - 1:-1].tolist()
            end_indexes = np.argsort(end_logit)[-1:-n_best - 1:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def flip_qa(model, tokenizer, context0, inputs, token_chunks, x, template, example, tokenized_example, fracs, flip_case,
            UNK_IDX=1, random_order=False, device='cpu', controls=None, filter_f1=True):
    """Performs the input reduction experiment for one sample"""

    PAD = tokenizer.pad_token_id
    tokens = list(itertools.chain(*token_chunks))

    token_chunks_dict = {i: chunk for i, chunk in enumerate(token_chunks)}

    # Compute standard forward pass
    inputs0 = torch.tensor(np.array(inputs['input_ids'])).to(device)
    attention_mask = torch.tensor(np.array(inputs['attention_mask'])).to(device)
    token_type_ids = torch.tensor(np.array(inputs['token_type_ids'])).to(device)

    y0 = model(input_ids=inputs0, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=False)

    start_logits = y0['start_logits'].detach().squeeze().cpu().numpy()
    end_logits = y0['end_logits'].detach().squeeze().cpu().numpy()
    offset_mapping = tokenized_example['offset_mapping'][0]

    predicted_answer = get_best_answer(start_logit=start_logits,
                                       end_logit=end_logits,
                                       offsets=offset_mapping,
                                       context=context0,
                                       example_id=0)

    logit_mask = predicted_answer["mask"]

    results_val = compute_metrics(start_logits[np.newaxis, :], end_logits[np.newaxis, :], tokenized_example, example)

    if filter_f1:
        if results_val['f1'] < 33.:
            print('skip')
            return None

    answer = example[0]['answers']['text'][0]  # example.answer[0]
    mask = get_tokenized_answer_mask(example[0]['answers']['text'][0], inputs['input_ids'][0], tokenizer)

    if mask is None:
        print('skip')

        return None

    if random_order == False:
        if flip_case == 'generate':
            inds_sorted = np.argsort(x)[::-1]
        elif flip_case == 'destroy':
            inds_sorted = np.argsort(np.abs(x))
        else:
            print('Select either "generate" or "destroy" reduction cases')
            raise
    else:
        inds_sorted = np.argsort(x)[::-1]
        np.random.shuffle(inds_sorted)

    vals = x[inds_sorted]

    y_true = None

    F1 = []
    evidence = []
    evidence_soft = []

    model_outs = {'sentence': None, 'y_true': y_true, 'y0': y0}

    N = len(x)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip = int(np.ceil(frac * N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]
        # inds_flip_spacy = [inds_map[str(i)] for i in inds_flip]

        if flip_case == 'destroy':
            token_chunks_dict_ = {k: v for k, v in token_chunks_dict.items()}
            for i in inds_flip:
                token_chunks_dict_[i] = [UNK_IDX] * len(token_chunks_dict[i])


        elif flip_case == 'generate':
            token_chunks_dict_ = {k: [UNK_IDX] * len(v) for k, v in token_chunks_dict.items()}
            for i in inds_flip:
                token_chunks_dict_[i] = token_chunks_dict[i]

        inputs = template + [[PAD] * int((inputs0.squeeze() == PAD).sum())]

        inputs[1] = list(itertools.chain(*[token_chunks_dict_[k] for k in sorted(token_chunks_dict_.keys())]))

        context = inputs[1] + inputs[2]

        inputs = torch.tensor(np.array(list(itertools.chain(*inputs)))).unsqueeze(0).to(device)

        if inputs.shape != inputs0.shape:
            print('not matched')
            return None

        y = model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)

        start_logits = y['start_logits'].detach().squeeze().cpu().numpy()

        end_logits = y['end_logits'].detach().squeeze().cpu().numpy()

        answer = y_true

        offset_mapping = tokenized_example['offset_mapping'][0]

        predicted_answer = get_best_answer(start_logit=start_logits,
                                           end_logit=end_logits,
                                           offsets=offset_mapping,
                                           context=context0,
                                           example_id=0)

        logit_mask = predicted_answer["mask"]

        #     import pdb;pdb.set_trace()

        results_val = compute_metrics(start_logits[np.newaxis, :], end_logits[np.newaxis, :], tokenized_example,
                                      example)

        print(frac, results_val)

        f1 = results_val['f1']
        exact_match = results_val['exact_match']

        # Evaluation

        p_end = softmax(end_logits)[mask == 3]
        p_start = softmax(start_logits)[mask == 1] if (mask == 1).sum() > 0 else p_end
        score = ((p_start + p_end) / 2.).squeeze()  # should be logit based...

        # fuzzy scoring
        mask_start = np.zeros_like(end_logits)
        mask_start[mask == 1] = 1.
        mask_start_gauss = gaussian_filter1d(mask_start, 1.)

        mask_end = np.zeros_like(end_logits)
        mask_end[mask == 3] = 1.
        mask_end_gauss = gaussian_filter1d(mask_end, 1.)

        #  p_end_weighted = (softmax(end_logits)*mask_end_gauss).sum()
        #  p_start_weighted = (softmax(start_logits)*mask_start_gauss if (mask==1).sum()>0 else p_end_weighted).sum()

        p_end_weighted = (softmax(end_logits)[mask_end_gauss > 0]).max()
        p_start_weighted = (softmax(start_logits)[mask_start_gauss > 0]).max() if (
                                                                                          mask == 1).sum() > 0 else p_end_weighted

        # import pdb;pdb.set_trace()
        score_soft = ((p_start_weighted + p_end_weighted) / 2.).squeeze()  # should be logit based...

        F1.append(f1)
        evidence.append(score)
        evidence_soft.append(score_soft)

        evolution[frac] = None  # (inputs.detach().cpu().numpy(), inds_flip, y)

        gc.collect()
        torch.cuda.empty_cache()

    if flip_case == 'generate' and frac == 1.:

        try:
            assert (inputs0 == inputs).all()
        except:
            diff0 = inputs0[inputs0 != inputs]
            diff1 = inputs[inputs0 != inputs]

            toks0 = tokenizer.convert_ids_to_tokens(diff0)
            toks1 = tokenizer.convert_ids_to_tokens(diff1)

            print(toks0, toks1)
        #    import pdb;pdb.set_trace()

    model_outs['flip_evolution'] = evolution
    return F1, evidence, evidence_soft, model_outs, predicted_answer


def compare_listcomp(x, y, ignore_ids=[]):
    x = [i for i in x if i not in ignore_ids]
    y = [i for i in y if i not in ignore_ids]

    return list(set(x).intersection(set(y)))


def flip_qa_pre(model, tokenizer, context0, inputs, token_chunks, x, template, example, tokenized_example, fracs,
                flip_case, UNK_IDX=1, random_order=False, device='cpu', controls=None, filter_f1=True, seed=1):
    """Performs the input reduction experiment for one sample"""

    PAD = tokenizer.pad_token_id
    tokens = list(itertools.chain(*token_chunks))

    token_chunks_dict = {i: chunk for i, chunk in enumerate(token_chunks)}

    # Compute standard forward pass
    inputs0 = torch.tensor(np.array(inputs['input_ids'])).to(device)
    attention_mask = torch.tensor(np.array(inputs['attention_mask'])).to(device)
    token_type_ids = torch.tensor(np.array(inputs['token_type_ids'])).to(device)

    y0 = model(input_ids=inputs0, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=False)

    start_logits = y0['start_logits'].detach().squeeze().cpu().numpy()
    end_logits = y0['end_logits'].detach().squeeze().cpu().numpy()
    offset_mapping = tokenized_example['offset_mapping'][0]

    predicted_answer = get_best_answer(start_logit=start_logits,
                                       end_logit=end_logits,
                                       offsets=offset_mapping,
                                       context=context0,
                                       example_id=0)

    logit_mask = predicted_answer["mask"]

    results_val = compute_metrics(start_logits[np.newaxis, :], end_logits[np.newaxis, :], tokenized_example, example)

    if filter_f1:
        if results_val['f1'] < 33.:
            print('skip')
            return None

    answer = example[0]['answers']['text'][0]  # example.answer[0]
    input_ids = inputs['input_ids'][0]
    mask = get_tokenized_answer_mask(answer, input_ids, tokenizer)

    if mask is None:
        print('skip')
        return None

    # Ignoring punctuation when matching answer words
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    punctuation_ids = list(itertools.chain(*[tokenizer(p)['input_ids'][1:-1] for p in punctuation]))

    answer_ids = np.array(input_ids)[mask > 0].tolist()

    answer_chunks = [(k, v) for k, v in token_chunks_dict.items() if
                     len(compare_listcomp(answer_ids, v, punctuation_ids)) > 0]
    answer_chunk_ids = [k[0] for k in answer_chunks]

    print(' '.join(tokenizer.convert_ids_to_tokens(answer_ids)))
    print(answer, '\n', [' '.join(tokenizer.convert_ids_to_tokens(k[1])) for k in answer_chunks])

    N = len(x)

    # Adding the inital tokens
    if fracs[0] > 0:
        init_inds = np.array(list(range(N)))

        init_inds = [ind for ind in init_inds if ind not in answer_chunk_ids]

        n_flip = int(np.ceil(fracs[0] * N))
        np.random.seed(seed)
        np.random.shuffle(init_inds)
        init_inds = init_inds[:n_flip]  # .tolist()
    else:
        init_inds = None

    if random_order == False:
        if flip_case == 'generate':
            inds_sorted = np.argsort(x)[::-1]
        elif flip_case == 'destroy':
            inds_sorted = np.argsort(np.abs(x))
        else:
            print('Select either "generate" or "destroy" reduction cases')
            raise
    else:
        inds_sorted = np.argsort(x)[::-1]
        np.random.shuffle(inds_sorted)

    if init_inds is not None:
        inds_sorted = np.array(init_inds + [ind for ind in inds_sorted.tolist() if ind not in init_inds])

    vals = x[inds_sorted]

    y_true = None

    F1 = []
    evidence = []
    evidence_soft = []

    model_outs = {'sentence': None, 'y_true': y_true, 'y0': y0}

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip = int(np.ceil(frac * N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'destroy':
            token_chunks_dict_ = {k: v for k, v in token_chunks_dict.items()}
            for i in inds_flip:
                token_chunks_dict_[i] = [UNK_IDX] * len(token_chunks_dict[i])


        elif flip_case == 'generate':
            token_chunks_dict_ = {k: [UNK_IDX] * len(v) for k, v in token_chunks_dict.items()}
            for i in inds_flip:
                token_chunks_dict_[i] = token_chunks_dict[i]

        inputs = template + [[PAD] * int((inputs0.squeeze() == PAD).sum())]

        inputs[1] = list(itertools.chain(*[token_chunks_dict_[k] for k in sorted(token_chunks_dict_.keys())]))

        input_ids_flip = inputs[1]

        context = inputs[1] + inputs[2]

        inputs = torch.tensor(np.array(list(itertools.chain(*inputs)))).unsqueeze(0).to(device)

        if inputs.shape != inputs0.shape:
            print('not matched')
            return None

        y = model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)

        start_logits = y['start_logits'].detach().squeeze().cpu().numpy()

        end_logits = y['end_logits'].detach().squeeze().cpu().numpy()

        answer = y_true

        offset_mapping = tokenized_example['offset_mapping'][0]

        predicted_answer = get_best_answer(start_logit=start_logits,
                                           end_logit=end_logits,
                                           offsets=offset_mapping,
                                           context=context0,
                                           example_id=0)

        logit_mask = predicted_answer["mask"]

        results_val = compute_metrics(start_logits[np.newaxis, :], end_logits[np.newaxis, :], tokenized_example,
                                      example)

        print(frac, results_val)

        f1 = results_val['f1']
        exact_match = results_val['exact_match']

        # Evaluation 
        # hard scoring only considerung the probability of the ground truth answer span
        p_end = softmax(end_logits)[mask == 3]
        p_start = softmax(start_logits)[mask == 1] if (mask == 1).sum() > 0 else p_end
        score = ((p_start + p_end) / 2.).squeeze()  # should be logit based...

        # fuzzy scoring - taking a span around the ground truth span and selecting the max
        mask_start = np.zeros_like(end_logits)
        mask_start[mask == 1] = 1.
        mask_start_gauss = gaussian_filter1d(mask_start, 1.)

        mask_end = np.zeros_like(end_logits)
        mask_end[mask == 3] = 1.
        mask_end_gauss = gaussian_filter1d(mask_end, 1.)

        p_end_weighted = (softmax(end_logits)[mask_end_gauss > 0]).max()
        p_start_weighted = (softmax(start_logits)[mask_start_gauss > 0]).max() if (
                                                                                          mask == 1).sum() > 0 else p_end_weighted

        score_soft = ((p_start_weighted + p_end_weighted) / 2.).squeeze()

        F1.append(f1)
        evidence.append(score)
        evidence_soft.append(score_soft)

        input_ids_ = [i for i in inputs.detach().squeeze().cpu().numpy().tolist() if i not in [PAD]]

        words = tokenizer.convert_ids_to_tokens(input_ids_)

        input_ids_ = ' '.join(words)

        #  print(frac, input_ids_.replace('[UNK]', '_' ))

        words_context = tokenizer.convert_ids_to_tokens(input_ids_flip)

        evolution[frac] = words_context

        gc.collect()
        torch.cuda.empty_cache()

    if flip_case == 'generate' and frac == 1.:

        try:
            assert (inputs0 == inputs).all()
        except:
            diff0 = inputs0[inputs0 != inputs]
            diff1 = inputs[inputs0 != inputs]

            toks0 = tokenizer.convert_ids_to_tokens(diff0)
            toks1 = tokenizer.convert_ids_to_tokens(diff1)

            print(toks0, toks1)

    evolution['init'] = (x, vals)
    #  import pdb;pdb.set_trace()

    model_outs['flip_evolution'] = evolution
    return F1, evidence, evidence_soft, model_outs, predicted_answer
