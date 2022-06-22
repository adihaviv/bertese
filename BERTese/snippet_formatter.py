import torch
import re

if not torch.cuda.is_available():
    from BERTese.utils.lama_utils import *
    from BERTese.utils.logging_utils import print_log_to_screen as print_log
    import BERTese.bert_utils as bert
    import BERTese.nlp_utils as nlp
else:
    from utils.lama_utils import *
    from utils.logging_utils import print_log_to_screen as print_log
    import bert_utils as bert
    import nlp_utils as nlp

EOMs = [",", "\\.", "\\)"]
MASK = bert.MASK[1:-1]


def full_masked_sentence(example, nee):
    return example.snippet, True


def identity(example, nee):
    return example.masked_sentence, True


# this is for sanity, to test the seq2seq model for more than identity
def rewrite_identity(example, nee):
    out = "it seems that " + example.masked_sentence
    if "birth" in example.relation.lower():
        out = out.replace("born", "raised")
    if "languages spoken" in example.relation.lower():
        out = out.replace("communicate in", "speak")
    if "death" in example.relation.lower():
        out = out.replace("died", "passed away")
    if "award" in example.relation.lower():
        out = out.replace("was awarded the", "received the award of")
    if "work location" in example.relation.lower():
        out = out.replace("used to work in", "work location was")
    if "instrument" in example.relation.lower():
        out = out.replace("plays", "plays the")
    if "instance" in example.relation.lower():
        out = out.replace("is a", "is a instance of")
    if "capital" in example.relation.lower():
        out = out.replace("capital", "the primary city")
    if "continent" in example.relation.lower():
        out = out.replace("located", "passed away")
    if "native language" in example.relation.lower():
        out = out.replace("native", "first")
    return out, True


def replace_subject_with_constant(example, nee):
    return _replace_subject(example.masked_sentence, example, "entity"), True


def replace_subject_with_relation(example, nee):
    return _replace_subject(example.masked_sentence, example, example.label_), True


def rewrite_replace_subject_with_constant(example, nee):
    return _replace_subject(rewrite_identity(example,nee)[0], example, "entity"), True


def rewrite_replace_subject_with_relation(example, nee):
    return _replace_subject(rewrite_identity(example, nee)[0], example, example.label_), True


def _replace_subject(input, example, replace_value):
    start = input.lower().find(example.sub_label.lower())
    res = input[:start] + replace_value + input[start + len(example.sub_label):]
    return res


def end_mask_sentence(example, nee):
    valid_for_train = True
    formatted_sentence = example.snippet
    start_idx = 0
    end_idx = 0
    subject = get_sub_label(example.snippet, example.sub_label)
    try:
        x_idxs = [(match.start(), match.end()) for match in re.finditer(subject.lower(), example.snippet.lower())]
    except Exception as e:
        print_log("\nException {}\n".format(e))
        print_log(example)
        print_log("\nsubject:{} \n snippet:{}\n".format(subject.lower(), example.snippet.lower()))
        return formatted_sentence

    if len(x_idxs) == 0:
        print_log("SUBJECT WAS NOT FOUND IN EVIDENCE:\nRelation:{}\nSubject:{}\nSnippet:{}\n".format(
            example.relation, subject, formatted_sentence))
        x_idxs = [(0, 1)]
        valid_for_train = False

    mask_idxs = [(match.start() - 1, match.end() + 1) for match in re.finditer(MASK, example.snippet)]

    for mask_idx in mask_idxs:
        for x_idx in x_idxs:
            if x_idx[1] < mask_idx[0]:
                start_idx = x_idx[0]
                end_idx = mask_idx[1]
                break
        if end_idx > 0:
            break

    if start_idx >= end_idx:
        start_idx = 0
        end_idx = len(formatted_sentence) - 1
        valid_for_train = False

    return _format(end_idx, formatted_sentence, start_idx, True), valid_for_train


def eos_mask_sentence(example, nee):
    valid_for_train = True
    formatted_sentence = example.snippet
    start_idx = 0
    end_idx = 0
    subject = get_sub_label(example.snippet, example.sub_label)

    try:
        x_idxs = [(match.start(), match.end()) for match in re.finditer(subject.lower(), example.snippet.lower())]
    except Exception as e:
        print_log("\nException {}\n".format(e))
        print_log(example)
        print_log("\nsubject:{} \n snippet:{}\n".format(subject.lower(), example.snippet.lower()))
        return formatted_sentence, False

    if len(x_idxs) == 0:
        print_log("SUBJECT WAS NOT FOUND IN EVIDENCE:\nRelation:{}\nSubject:{}\nSnippet:{}\n".format(
            example.relation, subject, formatted_sentence))
        x_idxs = [(0, 1)]
        valid_for_train = False

    mask_idxs = [(match.start()-1, match.end()+1) for match in re.finditer(MASK, example.snippet)]
    eom_idxs = []
    for eom in EOMs:
        eom_idxs = eom_idxs + [match.end() for match in re.finditer(eom, example.snippet)]
    if not len(example.snippet) in eom_idxs:
        eom_idxs.append(len(example.snippet))
    eom_idxs.sort()

    if len(eom_idxs) == 0:
        eom_idxs = [len(example.snippet)]

    if len(mask_idxs) == 0:
        print("\nMASK_IDX is None\n")
        print_log(example)
        valid_for_train = False
        return formatted_sentence, valid_for_train

    if len(x_idxs) == 0:
        print_log("\n X_IDX is None\n")
        print_log(example)
        valid_for_train = False
        return formatted_sentence, valid_for_train

    subject_before_mask = x_idxs[0][0] < mask_idxs[0][0]

    for mask_idx in mask_idxs:
        for x_idx in x_idxs:
            if subject_before_mask and x_idx[1] < mask_idx[0]:
                start_idx = x_idx[0]
                for eom_idx in eom_idxs:
                    if eom_idx >= mask_idx[1]:
                        end_idx = eom_idx
                        break
            elif not subject_before_mask and mask_idx[1] < x_idx[0]:
                start_idx = mask_idx[0]
                for eom_idx in eom_idxs:
                    if eom_idx > x_idx[1]:
                        end_idx = eom_idx
                        break
            if end_idx > 0:
                break
        if end_idx > 0:
            break

    if start_idx > end_idx:
        start_idx = 0
        end_idx = len(formatted_sentence) - 1
        valid_for_train = False

    return _format(end_idx, formatted_sentence, start_idx, False), valid_for_train


def end_with_mask_ner(example, nee):
    formatted_snippet, valid_for_train = end_mask_sentence(example)
    return _replace_entities(example, formatted_snippet, False, nee), valid_for_train


def end_with_eos_mask_ner(example, nee):
    formatted_snippet, valid_for_train = eos_mask_sentence(example)
    return _replace_entities(example, formatted_snippet, False, nee), valid_for_train


def end_with_mask_random_ner(example, nee):
    formatted_snippet, valid_for_train = end_mask_sentence(example, nee)
    return _replace_entities(example, formatted_snippet, True, nee), valid_for_train


def end_with_eos_mask_random_ner(example, nee):
    formatted_snippet, valid_for_train = eos_mask_sentence(example, nee)
    return _replace_entities(example, formatted_snippet, True, nee), valid_for_train


def _replace_entities(example, formatted_snippet, replace_with_random_value, nee):
    return _replace_entities_with_value(get_sub_label(example.snippet, example.sub_label),
                                        formatted_snippet, replace_with_random_value, nee,
                                        replace_with_random_value=replace_with_random_value)


def _replace_entities_with_value(source_entity, formatted_text, nee, replace_value=None, replace_with_random_value=False):
    subject = source_entity
    doc = nlp(formatted_text)
    offset = 0
    for ent in doc.ents:
        if len(ent.text) < 2 or MASK.lower() in ent.text.lower():
            continue
        ent_text = nlp.clear_text(ent.text)
        if ent_text.lower() == subject.lower():
            continue
        ent_end_char = ent.end_char - (len(ent.text) - len(ent_text))
        replace_value = replace_value if replace_value is not None else \
                        nee.get_random_entity(ent.label_) if replace_with_random_value else ent.label_
        res = formatted_text[:ent.start_char + offset] + replace_value + \
                            formatted_text[ent_end_char + offset:]
        offset -= len(ent_text) - len(replace_value)
    return res


def _format(end_idx, formatted_sentence, start_idx, cut_after_mask):
    start_with_prentice = "(" in formatted_sentence[start_idx:end_idx] and \
                          not ")" in formatted_sentence[start_idx:end_idx]
    if start_with_prentice:
        if cut_after_mask:
            formatted_sentence = formatted_sentence[start_idx:end_idx] + ")."
        elif not ")" in formatted_sentence[start_idx:]:
            formatted_sentence = formatted_sentence[start_idx:end_idx] + ")."
        else:
            pern_indices = [i for i, ltr in enumerate(formatted_sentence) if ltr == ")"]
            pern_indices.sort()
            for pern_idx in pern_indices:
                if pern_idx > start_idx and \
                   formatted_sentence[start_idx:pern_idx+1].count("(") == formatted_sentence[start_idx:pern_idx+1].count(")"):
                    end_idx = pern_idx
                    break
            formatted_sentence = formatted_sentence[start_idx:end_idx+1] + "."
    elif formatted_sentence[end_idx - 1] in [",", "."]:
        formatted_sentence = formatted_sentence[start_idx: end_idx - 1] + "."
    else:
        formatted_sentence = formatted_sentence[start_idx: end_idx] + "."
    return formatted_sentence
