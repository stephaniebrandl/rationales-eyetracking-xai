# Aligning the tokenization of different language model tokenizers with the tokenization in the eye-tracking corpora is really tricky.
# We did our best to account for as many cases as possible.
# Some cases are so specific that they would need to be hard-coded.
# For example, the ZUCO corpus contains a few instances of "U.S" which is seen as a single token but separated by most tokenizers.
# We decided to simply ignore these very specific cases but encourage you to do better.

import numpy as np
from scipy.special import softmax


def postprocess_attention(tokens, importance, lang):
    if lang == 'es' and '133[UNK]134:38-1[UNK]38-11' in "".join(tokens):
        return tokens[:-6], importance[:-6]
    elif lang == 'en' and '133[UNK]134:38-1[UNK]38-11' in "".join(tokens):
        return tokens[:-5], importance[:-5]
    elif lang == 'en' and '133–134:38-1–38-11' in "".join(tokens) and len(tokens) > 72:
        return tokens[:-5], importance[:-5]
    elif lang == 'es' and 'regional[UNK].' in "".join(tokens):
        return tokens[:-1], importance[:-1]
    else:
        return tokens, importance


def preprocess_attention(tokens, relative_attention, modelname, lang, text_id):
    tokens, merged_attention = merge_subwords(tokens, relative_attention)
    if any(model in modelname for model in ["mt5", 'xlmr', 'roberta']):
        begin_token = {"mt5": "▁", "xlmr": "▁", "xlmr_large": "▁", "roberta": "Ġ"}
        tokens, merged_attention = merge_albert_tokens(
            tokens, merged_attention, begin_token[modelname])
    tokens, merged_attention = merge_hyphens(tokens, merged_attention, lang, text_id)
    tokens, merged_attention = merge_symbols(tokens, merged_attention, lang)
    if lang != 'es':
        tokens, merged_attention = merge_numeric(tokens, merged_attention)
        # tokens, merged_attention = merge_numeric(tokens, merged_attention)
    tokens, merged_attention = merge_special_cases(tokens, merged_attention)
    tokens, merged_attention = merge_special_cases(tokens, merged_attention)
    tokens, merged_attention = merge_UNK(tokens, merged_attention)
    tokens, merged_attention = merge_UNK(tokens, merged_attention)
    tokens, merged_attention = remove_UNK(tokens, merged_attention, text_id)
    return tokens, merged_attention


def merge_subwords(tokens, summed_importance, pooling='max'):
    adjusted_tokens = []
    adjusted_importance = []

    current_token = ""
    current_importance = 0

    # Tokenizers use different word piece separators. We simply check for both here
    word_piece_separators = ("##", "_")
    for i, token in enumerate(tokens):
        # We sum the importance of word pieces
        if pooling == 'max':
            current_importance = current_importance if current_importance > summed_importance[i] else summed_importance[
                i]
        else:
            current_importance += summed_importance[i]

        # Identify word piece
        if token.startswith(word_piece_separators):
            # skip the hash tags
            current_token += token[2:]

        else:
            current_token += token

        # Is this the last token of the sentence?
        if i == len(tokens) - 1:
            adjusted_tokens.append(current_token)
            adjusted_importance.append(current_importance)

        else:
            # Are we at the end of a word?
            if not tokens[i + 1].startswith(word_piece_separators):
                # append merged token and importance
                adjusted_tokens.append(current_token)
                adjusted_importance.append(current_importance)

                # reset
                current_token = ""
                current_importance = 0
    return adjusted_tokens, adjusted_importance


# Word piece tokenization splits words separated by hyphens. Most eye-tracking corpora don't do this.
# This method sums the importance for tokens separated by hyphens.
def merge_hyphens(tokens, importance, lang, text_id, pooling='max'):
    adjusted_tokens = []
    adjusted_importance = []

    if "-" in tokens or "'" in tokens:
        # Get all indices of -
        indices = [i for i, x in enumerate(tokens) if ((x == "-") or (x == "'" and tokens[i + 1] == "s"))
                   and not (lang == 'de' and tokens[i + 1] == 'Delta' and i in [2, 4])
                   and not 'φορος' in tokens[i + 1]]

        i = 0
        while i < len(tokens):
            if i + 1 in indices and i + 2 < len(tokens):
                if lang == 'de' and i in [64, 98]:
                    combined_token = tokens[i] + tokens[i + 1]
                    if pooling == 'max':
                        combined_heat = np.max([importance[i], importance[i + 1]])
                    else:
                        combined_heat = importance[i] + importance[i + 1]
                    i += 2
                else:
                    combined_token = tokens[i] + tokens[i + 1] + tokens[i + 2]
                    if pooling == 'max':
                        combined_heat = np.max([importance[i], importance[i + 1], importance[i + 2]])
                    else:
                        combined_heat = importance[i] + importance[i + 1] + importance[i + 2]
                    i += 3
                adjusted_tokens.append(combined_token)
                adjusted_importance.append(combined_heat)
            elif i in indices:
                adjusted_tokens[-1] += tokens[i] + tokens[i + 1]
                if pooling == 'max':
                    adjusted_importance[-1] = np.max([adjusted_importance[-1], importance[i], importance[i + 1]])
                else:
                    adjusted_importance[-1] += importance[i] + importance[i + 1]

                i += 2
            else:
                adjusted_tokens.append(tokens[i])
                adjusted_importance.append(importance[i])
                i += 1
        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


# Word piece tokenization splits parentheses and currency symbols as separate tokens. This is not done in Zuco.

def merge_symbols(tokens, importance, lang, pooling='max'):
    initial_symbols = ["(", "$", "€", "\"", "\'", "[", '("', "„", "(«", "«", "~", '-']
    if lang in ['de', 'es']:
        end_symbols = [")", "\"", "\'", ",", ".", "]", "+", ";", "/", ":", "'", "re", '"),', ').', "»),", "»", "),"]
    else:
        end_symbols = [")", "%", "\"", "\'", ",", ".", "]", "+", ";", "/", ":", "'", "re", '"),', ').', "»),", "»",
                       "),", "t", '".']

    all_symbols = initial_symbols + end_symbols
    # First check if anything needs to be done
    while any(token in all_symbols for token in tokens):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens) - 1:
            combined_token = tokens[i]
            combined_heat = importance[i]

            # Nothing to be done for the last token
            if i <= len(tokens) - 2:
                # Glue the parentheses back to the token
                while tokens[i] in initial_symbols:
                    combined_token = combined_token + tokens[i + 1]
                    if pooling == 'max':
                        combined_heat = np.max([combined_heat, importance[i + 1]])
                    else:
                        combined_heat = combined_heat + importance[i + 1]
                    i += 1

                while i < len(tokens) - 1 and tokens[i + 1] in end_symbols:
                    # and not(lang=='es' and tokens[i] in ['20', '40']):
                    combined_token = combined_token + tokens[i + 1]
                    if pooling == 'max':
                        combined_heat = np.max([combined_heat, importance[i + 1]])
                    else:
                        combined_heat = combined_heat + importance[i + 1]
                    i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


# tokens = ['[', '(', '1979', ')', '.', 'random']
# importance = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# lang = 'en'
# adjusted_tokens, adjusted_importance = merge_symbols(tokens, importance, lang)
# print(adjusted_tokens, adjusted_importance)


def merge_numeric(tokens, importance, pooling='max'):
    if any(char.isnumeric() for token in tokens for char in token):
        exceptions = ['km2', '1974', '1754', '1967', '1⁄2', 'Internet2', '(145', '21-acre']
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens) - 1:
            combined_token = tokens[i]
            combined_heat = importance[i]
            n_numeric_i = np.sum([char.isnumeric() for char in tokens[i]])
            n_numeric_ii = np.sum([char.isnumeric() for char in tokens[i + 1]]) if i + 1 < len(tokens) else 0
            while i < len(tokens) - 1 and any(char.isnumeric() for char in tokens[i]) \
                    and any(char.isnumeric() for char in tokens[i + 1]) \
                    and not n_numeric_i == len(tokens[i]) == 4 \
                    and not n_numeric_ii == len(tokens[i + 1]) == 4 \
                    and not (any(tok in tokens[i] for tok in exceptions)) \
                    and not (any(tok in tokens[i + 1] for tok in exceptions)):
                combined_token = combined_token + tokens[i + 1]
                if pooling == 'max':
                    combined_heat = np.max([combined_heat, importance[i + 1]])
                else:
                    combined_heat = combined_heat + importance[i + 1]
                i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1
        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


def merge_special_cases(tokens, importance, pooling='max'):
    special_cases = ['L.P.', '(e.g.,', 'Pertwee/Tom', 'pertwee/tom', 'compounding/dispensing', 'raise"),', 'Gbit/s',
                     'Gbit/s.',
                     '(d.h.', '4.a', '2002;,Robert', '5.a', '1,5', '5,3', '51,6', '32,9', '0,62', '0,37', '1,1',
                     '27/100.',
                     'Rhein-Maas-', '{0,1', '{0,1})', 'chimenea...).', 'compresiones.:133–134:38-1–38-11', 'regional”.',
                     'V&', 'V&A', 'v&', 'v&a', 'l.p.']
    if any(case in "".join(tokens) for case in special_cases):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens) - 1:
            combined_token = tokens[i]
            combined_heat = importance[i]
            if i < len(tokens) - 1 and tokens[i] + tokens[i + 1] in special_cases:
                combined_token = combined_token + tokens[i + 1]
                if pooling == 'max':
                    combined_heat = np.max([combined_heat, importance[i + 1]])
                else:
                    combined_heat = combined_heat + importance[i + 1]
                i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1
        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


def merge_UNK(tokens, importance, pooling='max'):
    special_cases = ['1986[UNK]when', 'algae[UNK]glaucophytes,', 'plants[UNK]in', 'Geelong[UNK]will',
                     'lineages[UNK]the', 'cyanobacterium[UNK]which', 'true[UNK]both', 'ecosystems[UNK]of',
                     'Rhine[UNK]Meuse', 'Rhine-Meuse[UNK]Scheldt', 'Elway[UNK]mánager', 'hinduismo[UNK]especialmente',
                     '29,000[UNK]24,000', '2001[UNK]02', '(1185[UNK]1226),', '23[UNK]16,', '20[UNK]18,',
                     '1986—when', 'algae—glaucophytes,', 'plants—in', 'geelong—will', 'Geelong—will',
                     'lineages—the', 'cyanobacterium—which', 'true—both', 'ecosystems—of',
                     'Rhine—Meuse', 'Meuse–Scheldt', 'Rhine–Meuse-Scheldt',
                     'rhine—meuse', 'meuse–scheldt', 'rhine–meuse-scheldt',
                     '29,000–24,000', '2001–02', '(1185–1226),', '23–16,', '20–18,']
    if any(case in "".join(tokens) for case in special_cases):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens) - 1:
            combined_token = tokens[i]
            combined_heat = importance[i]
            while i < len(tokens) - 2 and tokens[i] + tokens[i + 1] + tokens[i + 2] in special_cases:
                combined_token = combined_token + '-' + tokens[i + 2]
                if pooling == 'max':
                    combined_heat = np.max([combined_heat, importance[i + 2]])
                else:
                    combined_heat = combined_heat + importance[i + 2]
                i += 2
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1
        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


def remove_UNK(tokens, importance, text_id):
    special_cases = ['Klassen[UNK]', 'Privilegien[UNK]', 'Side[UNK]', 'Fresno[UNK]',
                     'War[UNK],', 'Bettler[UNK],', 'Polen)[UNK]']
    if any(case in "".join(tokens) for case in special_cases):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens) - 1:
            combined_token = tokens[i]
            combined_heat = importance[i]
            if i < len(tokens) - 2 and tokens[i] + tokens[i + 1] in special_cases:
                i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1
        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


def merge_albert_tokens(tokens, importance, begin_token, pooling='max'):
    adjusted_tokens = []
    adjusted_importance = []
    i = 0
    # We ignore the last token [SEP]
    while i < len(tokens) - 1:
        combined_token = tokens[i]
        combined_heat = importance[i]
        # Nothing to be done for the last token
        if i < (len(tokens) - 2):
            while not tokens[i + 1].startswith(begin_token) and not tokens[i] == '<s>':
                combined_token = combined_token + tokens[i + 1]
                if pooling == 'max':
                    combined_heat = np.max([combined_heat, importance[i + 1]])
                else:
                    combined_heat = combined_heat + importance[i + 1]
                i += 1
                if i == len(tokens) - 2:
                    break
        adjusted_tokens.append(combined_token.replace(begin_token, ""))
        adjusted_importance.append(combined_heat)
        i += 1
    # Add the last token
    adjusted_tokens.append(tokens[i])
    adjusted_importance.append(importance[i])
    return adjusted_tokens, adjusted_importance


# For the attention baseline, we fixed several experimental choices (see below) which might affect the results.
def calculate_relative_importance(tokens, importance, sep_token, pad_token, text_id,
                                  special_tokens=['[PAD]', '[CLS]', '[SEP]', '</s>', '<s>', '<pad>']):
    # 1. We take the mean over the 12 attention heads (like Abnar & Zuidema 2020)
    # I also tried the sum once, but the result was even worse
    # remove question
    if len(tokens) > len(importance):
        tokens = [tok for tok in tokens if tok != pad_token]
        assert (len(tokens) == len(importance))
    sep_index = tokens.index(sep_token)
    tokens = tokens[sep_index + 1:]
    mean_importance = importance[sep_index + 1:]

    tokens_out = []
    index_del = []

    for (itok, tok) in enumerate(tokens):
        if tok in special_tokens:
            index_del.append(itok)
        else:
            tokens_out.append(tok)

    mean_importance = np.delete(mean_importance, index_del, 0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    relative_attention = softmax(mean_importance)

    return tokens_out, relative_attention
