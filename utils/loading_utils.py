import numpy as np


def fixing_bugs(subdf, text_id, lang):
    try:
        if text_id == 'a_Amazonrainforest_4' and lang == 'en' and 'atypical' in subdf.word_id.tolist():
            idx = subdf.query("word_id=='atypical'").index[0]
            return split_rows(subdf, idx, 'a', 'typical')
        elif text_id == 'a_NikolaTesla_1' and lang == 'en' and 'Tesla_2' in subdf.word_id_orig.tolist():
            idx = subdf.query("word_id_orig=='Tesla_2'").index[0]
            return subdf.drop(index=idx)
        elif text_id == 'a_Construction_0' and lang == 'en' and 'planningcitation_0' in subdf.word_id_orig.tolist():
            idx = subdf.query("word_id_orig=='planningcitation_0'").index[0]
            return split_rows(subdf, idx, 'planning', 'citation')
        elif text_id in ['a_IntergovernmentalPanelonClimateChange_0', 'a_Teacher_4'
                                                                      'a_AmericanBroadcastingCompany_0',
                         'a_Apolloprogram_2', 'a_Kenya_0', 'a_VictoriaAustralia_2']:
            subdf = subdf[subdf.word_id != '—']
            return subdf[subdf.word_id != '=']
        elif text_id in ['a_GenghisKhan_0'] and 'son,_0' in subdf.word_id_orig.tolist():
            idx = subdf.query("word_id_orig=='son,_0'").index[0]
            return insert_row(subdf, idx, 'Jochi')
        elif text_id in ['a_SouthernCalifornia_2'] and 'second-_0' in subdf.word_id_orig.tolist():
            idx = subdf.query("word_id_orig=='second-_0'").index[0]
            return merge_rows(subdf, idx, idx + 1)
        elif text_id in ['a_Rhine_2', 'a_AmericanBroadcastingCompany_0']:
            return subdf[subdf.word_id_orig != '=_0']
        elif text_id == 'a_Huguenot_0' and 'century[citation' in subdf.word_id.tolist():
            idx = subdf.query("word_id=='century[citation'").index[0]
            return split_rows(subdf, idx, 'century', '[citation')
        elif text_id == 'a_Steamengine_3' and 'kWh[not' in subdf.word_id.tolist():
            idx = subdf.query("word_id=='kWh[not'").index[0]
            return split_rows(subdf, idx, 'kWh', '[not')
        elif text_id == 'a_Packetswitching_2' and lang == 'en' and 'seven-' in subdf.word_id.tolist():
            idx = subdf.query("word_id=='seven-'").index[0]
            return merge_rows(subdf, idx, idx + 1)
        elif text_id in ['a_Computationalcomplexitytheory_0', 'a_Oxygen_1', 'a_Amazonrainforest_2', 'a_MartinLuther_3',
                         'a_Geology_2', 'a_Privateschool_0']:
            subdf = subdf[subdf.word_id != '—']
            return subdf[subdf.word_id != ' ']
        elif text_id in ['a_Islamism_0', 'a_VictoriaAustralia_1', 'a_Force_4']:
            return subdf[subdf.word_id != '|']
        elif text_id == 'a_Rhine_3' and lang == 'en' and subdf.loc[121, 'word_id'] == 'fall-':
            return merge_rows(subdf[subdf.word_id != '='], 121, 122)
        elif text_id == 'a_VictoriaAustralia_3' and lang == 'de':
            subdf = subdf[subdf.word_id != '—']
            return subdf[subdf.word_id != '=']
        elif text_id == 'a_VictoriaAustralia_3' and lang == 'en' and 'government-' in subdf.word_id.tolist():
            idx = subdf.query("word_id=='government-'").index[0]
            return merge_rows(subdf, idx, idx + 1)
        elif text_id == 'a_Packetswitching_4' and lang == 'en' and subdf.loc[133, 'word_id'] == 'investor':
            # import pdb;pdb.set_trace()
            # return insert_row(subdf, 133, 'in')
            return subdf
        elif text_id == 'a_SuperBowl50_1' and lang == 'es' and 'un_intento_0' in subdf.word_id_orig.tolist():
            idx = subdf.query("word_id_orig=='un_intento_0'").index[0]
            return split_rows(subdf, idx, 'un', 'intento')
        elif text_id == 'a_NikolaTesla_1' and lang == 'es' and 'accedieron' in subdf.word_id.tolist():
            idx = subdf.query("word_id=='accedieron'").index[0]
            df_out = insert_row(insert_row(subdf, idx, 'a'), 23, 'financiar')
            return df_out
        elif text_id == 'a_Computationalcomplexitytheory_1' and lang == 'es' and ' cddigo' in subdf.word_id.tolist():
            idx = subdf.query("word_id==' cddigo'").index[0]
            return insert_row(subdf, idx, 'binario.')
        elif text_id == 'a_NikolaTesla_4' and lang == 'de' and subdf.loc[23, 'word_id'] == 'Tesla':
            return insert_row(subdf, 23, 'beispielsweise.')
        elif text_id == 'a_Computationalcomplexitytheory_1' and lang == 'de' and subdf.loc[232, 'word_id'] == 'kénnen':
            return insert_row(subdf, 232, 'beispielsweise')
        else:
            return subdf
    except:
        import pdb;
        pdb.set_trace()


def merge_rows(df, index0, index1):
    df.loc[index0] = [df.loc[index0, "Unnamed: 0"],
                      np.nansum([df.loc[index0, "relFix"], df.loc[index1, "relFix"]]),
                      np.nansum([df.loc[index0, "TRT"], df.loc[index1, "TRT"]]),
                      df.loc[index0, "countFix"] + df.loc[index1, "countFix"],
                      df.loc[index0, "word_id"] + df.loc[index1, "word_id"],
                      df.loc[index0, "word_id_orig"][:-2] + df.loc[index1, "word_id_orig"],
                      df.loc[index0, "text_id"], df.loc[index0, "sentence_id"],
                      df.loc[index0, "word_length"] + df.loc[index1, "word_length"]]
    # df.drop(index=index1, inplace=True)
    df.drop(index=index1)
    return df


def split_rows(df, index0, word0, word1):
    df.loc[index0] = [df.loc[index0, "Unnamed: 0"],
                      df.loc[index0, "relFix"] / 2,
                      df.loc[index0, "TRT"] / 2,
                      df.loc[index0, "countFix"] / 2,
                      word0,
                      word0,
                      df.loc[index0, "text_id"],
                      df.loc[index0, "sentence_id"],
                      len(word0)]

    df.loc[index0 + 0.5] = [df.loc[index0, "Unnamed: 0"],
                            df.loc[index0, "relFix"],
                            df.loc[index0, "TRT"],
                            df.loc[index0, "countFix"],
                            word1,
                            word1,
                            df.loc[index0, "text_id"],
                            df.loc[index0, "sentence_id"],
                            len(word1)]
    return df.sort_index().reset_index(drop=True)


def insert_row(df, index0, word0):
    df.loc[index0 + 0.5] = [df.loc[index0, "Unnamed: 0"],
                            0,
                            0,
                            0,
                            word0,
                            word0,
                            df.loc[index0, "text_id"],
                            df.loc[index0, "sentence_id"],
                            len(word0)]

    return df.sort_index().reset_index(drop=True)
