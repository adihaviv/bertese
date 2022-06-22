import numpy as np

MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK_DUMMY = "mm"
CLS_DUMMY = "cc"
SEP_DUMMY = "ss"

PED_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
SPECIAL_IDS = [PED_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID]

UNUSED_IDS = [i for i in np.arange(998) if i not in SPECIAL_IDS]


def bert_encode(s, tokenizer):
    # for reason i don't know what is it the [CLS] token need to appears twice
    dummy_s = [CLS] + [CLS] + tokenizer.tokenize(s.replace(MASK, MASK_DUMMY).replace(SEP, SEP_DUMMY).replace(CLS, CLS_DUMMY)) + [SEP]
    dummy_s = [MASK if x == MASK_DUMMY else x for x in dummy_s]
    dummy_s = [SEP if x == SEP_DUMMY else x for x in dummy_s]
    dummy_s = [CLS if x == CLS_DUMMY else x for x in dummy_s]
    return tokenizer.convert_tokens_to_ids(dummy_s)


def bert_decode_clean(arr_ids, tokenizer, filter_unused=True, trim_cls_sep = False):
    if trim_cls_sep:
        arr_ids_no_cls = arr_ids[1:]
        return tokenizer.decode(
            arr_ids_no_cls[1:np.where(arr_ids_no_cls == SEP_ID)[0][0]] if SEP_ID in arr_ids_no_cls else arr_ids_no_cls[arr_ids_no_cls != PED_ID])

    trim_sep_or_remove_ped = arr_ids[:np.where(arr_ids == SEP_ID)[0][0] + 1] \
        if SEP_ID in arr_ids \
        else arr_ids[arr_ids!=PED_ID]

    filtered_ids, valid = _filter_garbage(trim_sep_or_remove_ped, filter_unused)
    res = ""
    if len(filtered_ids) > 0:
        res = tokenizer.decode(trim_sep_or_remove_ped)
    if not valid:
        res += "---> Trimmed, Many Tokens are [unused]"

    return res


def _filter_garbage(ids, filter_unused):
    if filter_unused:
        no_unused_ids = [i for i in ids if i not in UNUSED_IDS]
        if len(ids) - len(no_unused_ids) < 20: #if we have less than 20 unknown tokens we consider it to be valid
            return ids, True #todo: might be better to trim at the first unknown
        else:
            return no_unused_ids, False
    else:
        return ids, True


def get_mask_id(b_input_ids):
    #todo: implement
    return []