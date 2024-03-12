import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from utils.loading_utils import fixing_bugs
from utils.tokenization_util import preprocess_attention, postprocess_attention, calculate_relative_importance
from utils.utils import get_modelname
import click
import warnings
import yaml

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@click.command()
@click.option('--lang', default='en')
@click.option("--correct_answers", is_flag=True)
@click.option("--filter_quality", is_flag=True)
@click.option("--filter_vision", is_flag=True)
@click.option('--threshold', default=0.25)
@click.option('--vision', default="glasses")
@click.option('--workers', default="mturk")
def main(lang, correct_answers, filter_quality, threshold, filter_vision, vision, workers):
    data_dir = config["data_dir"]
    data_dir = data_dir + '_lab' if workers == 'lab' else data_dir
    data_dir = data_dir + '_volunteer' if workers == 'volunteer' else data_dir

    model_folder = config['model_dir']
    rationales_folder = config['rationale_dir']

    if filter_quality and threshold is not None:
        data_dir = data_dir + '_' + str(int(threshold))
    elif filter_quality and threshold is None:
        warnings.warn("filter_quality is set to True but no threshold is specified, "
                      "will continue without ignoring filter_quality")

    all_modelnames = ['mbert', 'distil', 'xlmr', 'xlmr_large']

    for modelbase in all_modelnames:
        for imodel in np.arange(3, 6):
            modelname = get_modelname(modelbase, lang, imodel)

            pad_token = '<pad>' if any(x in modelbase for x in ['xlmr', 'roberta']) else '[PAD]'
            sep_token = '</s>' if any(x in modelbase for x in ['xlmr', 'roberta']) else '[SEP]'
            special_tokens = ['[PAD]', '[CLS]', '[SEP]', '</s>', '<s>', '<pad>', 'Ġ']

            df_attention = pd.read_pickle(f'{config["model_dir"]}/{modelname}/attention.pkl')
            df_relevance = pd.read_pickle(f'{config["model_dir"]}/{modelname}/relevance.pkl')
            df_gi = pd.read_pickle(f'{config["model_dir"]}/{modelname}/gradientsinput.pkl')

            attention_columns = [col for col in df_attention if col.startswith('attention')]
            rollout_columns = [col for col in df_attention if col.startswith('rollout')]
            df_match = pd.read_pickle(f'./utils/mapping_id_qa_{lang}.pkl')
            dict_mapping = pd.Series(df_match.id.values, index=df_match.questions).to_dict()

            files = [file for file in listdir(rationales_folder) if
                     file.startswith('rationales') and lang.upper() in file]
            for file in sorted(files):
                df_rationales = pd.read_pickle(join(rationales_folder, file)).set_index('text_id')
                if modelbase == 'bert' and imodel == 3:
                    print(len(df_rationales), 'loading')
                for col in attention_columns:
                    df_rationales[col] = None
                for col in rollout_columns:
                    df_rationales[col] = None
                df_rationales['eyetracking_cont'] = None
                df_rationales['eyetracking_NR'] = None
                df_rationales['tok_et'] = None
                df_rationales['relevance'] = None
                df_rationales['gradientsinput'] = None
                df_rationales['tok_rel'] = None
                df_rationales['tok_att'] = None
                df_rationales['model_f1'] = None
                subject = file[11:-4]
                experiment = file.split('.')[0].split('_')[-1]

                filename = f"{subject}-relfix-feats_avg"
                filename = filename + '_correct' if correct_answers else filename
                filename = filename + f'_vision_{vision}' if filter_vision else filename

                try:
                    df_eyetracking = pd.read_csv(join(data_dir, filename + '.csv'))
                except FileNotFoundError:
                    continue

                df_eyetracking = df_eyetracking[df_eyetracking.word_id != " "]
                df_eyetracking = df_eyetracking.dropna(subset=['word_id'])
                for text_id, subdf in df_eyetracking.groupby('text_id'):
                    question_id = dict_mapping[df_rationales.loc[text_id, 'question_id']]

                    rel = df_relevance[df_relevance['id'] == question_id]['attention'].tolist()[0]
                    tok_rel = df_relevance[df_relevance['id'] == question_id]['tokens'].tolist()[0]

                    gi = df_gi[df_gi['id'] == question_id]['attention'].tolist()[0]
                    tok_gi = df_gi[df_gi['id'] == question_id]['tokens'].tolist()[0]
                    assert (all(token1 == token2 for token1, token2 in zip(tok_gi, tok_rel)))

                    if any(x in modelbase for x in ['xlmr', 'roberta']):
                        if '[' in tok_gi:
                            tok_gi[tok_gi.index('[')] = '▁[' if 'xlmr' in modelbase else 'Ġ['
                        elif ',[' in tok_gi:
                            tok_gi[tok_gi.index(',[')] = 'Ġ['

                    if any(x in modelbase for x in ['xlmr', 'roberta']):
                        if '[' in tok_rel:
                            tok_rel[tok_rel.index('[')] = '▁[' if 'xlmr' in modelbase else 'Ġ['
                        elif ',[' in tok_rel:
                            tok_rel[tok_rel.index(',[')] = 'Ġ['

                    tok_rel, rel = calculate_relative_importance(tok_rel, rel, sep_token, pad_token, text_id)
                    rel = [a for t, a in zip(tok_rel, rel) if t not in special_tokens]

                    tok_gi, gi = calculate_relative_importance(tok_gi, gi, sep_token, pad_token, text_id)
                    gi = [a for t, a in zip(tok_gi, gi) if t not in special_tokens]
                    tok_gi = [t for t in tok_gi if t not in special_tokens]
                    tok_gi, gi = preprocess_attention(tok_gi, gi, modelbase, lang, text_id)
                    tok_gi, gi = postprocess_attention(tok_gi, gi, lang)
                    df_rationales.at[text_id, 'gradientsinput'] = gi

                    tok_rel = [t for t in tok_rel if t not in special_tokens]
                    tok_rel, rel = preprocess_attention(tok_rel, rel, modelbase, lang, text_id)
                    tok_rel, rel = postprocess_attention(tok_rel, rel, lang)
                    df_rationales.at[text_id, 'relevance'] = rel
                    df_rationales.at[text_id, 'tok_rel'] = tok_rel

                    for col in rollout_columns:
                        att = df_attention[df_attention['id'] == question_id][col].tolist()[0]
                        tok = df_attention[df_attention['id'] == question_id]['tokens'].tolist()[0]
                        if any(x in modelbase for x in ['xlmr', 'roberta']):
                            if '[' in tok:
                                tok[tok.index('[')] = '▁[' if 'xlmr' in modelbase else 'Ġ['
                            elif ',[' in tok:
                                tok[tok.index(',[')] = 'Ġ['
                        att = [a for t, a in zip(tok, att) if t not in special_tokens]
                        tok = [t for t in tok if t not in special_tokens]
                        tok, att = preprocess_attention(tok, att, modelbase, lang, text_id)
                        tok, att = postprocess_attention(tok, att, lang)
                        df_rationales.at[text_id, col] = att

                    for col in attention_columns:
                        att = df_attention[df_attention['id'] == question_id][col].tolist()[0]
                        tok = df_attention[df_attention['id'] == question_id]['tokens'].tolist()[0]
                        if any(x in modelbase for x in ['xlmr', 'roberta']):
                            if '[' in tok:
                                tok[tok.index('[')] = '▁[' if 'xlmr' in modelbase else 'Ġ['
                            elif ',[' in tok:
                                tok[tok.index(',[')] = 'Ġ['
                        att = [a for t, a in zip(tok, att) if t not in special_tokens]
                        tok = [t for t in tok if t not in special_tokens]
                        tok, att = preprocess_attention(tok, att, modelbase, lang, text_id)
                        tok, att = postprocess_attention(tok, att, lang)
                        df_rationales.at[text_id, col] = att

                    df_rationales.at[text_id, 'tok_att'] = tok
                    df_rationales.at[text_id, 'model_f1'] = df_attention[
                        df_attention['id'] == question_id]['model'].values[0]

                    subdf = fixing_bugs(subdf, text_id, lang)

                    df_rationales.at[text_id, 'eyetracking_cont'] = subdf['relFix'].tolist()
                    df_rationales.at[text_id, 'tok_et'] = subdf['word_id'].tolist()

                filename_out = join(rationales_folder, f"{modelname}_{experiment}")
                filename_out = filename_out + '_lab' if workers == 'lab' else filename_out
                filename_out = filename_out + '_volunteer' if workers == 'volunteer' else filename_out
                filename_out = filename_out + '_correct-answers' if correct_answers else filename_out
                filename_out = filename_out + f'_vision_{vision}' if filter_vision else filename_out

                if filter_quality and threshold is not None:
                    filename_out = filename_out + '_' + str(threshold)

                if len(df_rationales) > 0:
                    if modelbase == 'bert' and imodel == 3:
                        print(len(df_rationales), 'saving')
                    df_rationales.to_pickle(filename_out + '.pkl')


if __name__ == '__main__':
    main()
