import json
import torch
import os
import pickle

if not torch.cuda.is_available():
    from BERTese.utils.lama_utils import LamaExample
    from BERTese.bert_utils import MASK
else:
    from utils.lama_utils import LamaExample
    from bert_utils import MASK


def load_file(args, filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if args.debug and args.items_for_debug <= len(data):
                break
            data.append(json.loads(line))

    return data


# GoogleRE
def get_google_re_parameters(args):
    relations = [
        {"relation": "place_of_birth", "template": "[X] was born in [Y] ."},
        {"relation": "date_of_birth", "template": "[X] (born [Y])."},
        {"relation": "place_of_death", "template": "[X] died in [Y] ."},
    ]
    if args.debug:
        relations = [
            {"relation": "place_of_birth_debug", "template": "[X] was born in [Y] ."}
        ]

    data_path_pre = args.lama_data_path +"/Google_RE/"
    data_path_post = "_test.jsonl"

    return relations, data_path_pre, data_path_post


def read_google_re_relation(args, filename, relation, masked_template):
    data = load_file(args, filename)

    lama_examples = []
    for sample in data:
        id = sample["uuid"]
        label = sample["obj_label"]
        sub_label = sample["sub_label"]
        snippet = sample["masked_sentences"][0]
        if len(sample["masked_sentences"]) > 1:
            if args.debug:
                print("--- more than 1 masked relations! for {} ---:".format(id))
                for masked_sentence in sample["masked_sentences"]:
                    try:
                        print(masked_sentence.encode('utf-8').decode("utf-8"))
                    except Exception as e:
                        print("encoding issue. check the files")
                    if MASK in masked_sentence:
                        snippet = masked_sentence
                        break

        new_masked_sentence = masked_template.replace("[X]", sub_label).replace("[Y]", MASK)
        lama_example = LamaExample(uuid=id, source="GoogleRE", relation=relation, masked_template=masked_template,
                                   snippet=snippet, masked_sentence=new_masked_sentence,
                                   sub_label=sub_label, label=label)
        lama_examples.append(lama_example)
    return lama_examples


def get_google_re_data(args):
    lama_examples = {}
    relations, data_path_pre, data_path_post = get_google_re_parameters(args)
    for relation in relations:
        relation_lama_examples = \
            read_google_re_relation(args, data_path_pre+relation["relation"]+data_path_post, relation["relation"], relation["template"])

        lama_examples[relation["relation"]] = (relation_lama_examples, relation["template"])
    return lama_examples


# T-Rex
def get_trex_parameters(args, data_folder="TREx"):
    data_path_pre = args.lama_data_path
    #relations = [
    #    {"relation": "test", "template": "[X] used to communicate in [Y] .", "label": "communication"}]
    relations = load_file(args, "{}relations.jsonl".format(data_path_pre))
    if args.debug:
        print("RUNNING IN TREX DEBUG MODE\n")
        relations = load_file(args, "{}relations_test.jsonl".format(data_path_pre))
    data_path_pre += data_folder+"/"
    data_path_post = ".jsonl"

    return relations, data_path_pre, data_path_post


def read_trex_relation(args, filename, relation, masked_template):
    data = load_file(args, filename)

    lama_examples = []
    for sample in data:
        id = sample["uuid"]
        label = sample["obj_label"]
        sub_label = sample["sub_label"]
        if len(sample["evidences"]) > 1:
            same = True
            prev_sub_label = ""
            prev_label = ""
            for evidence in sample["evidences"]:
                if len(prev_sub_label) > 0 and not(prev_sub_label == evidence["sub_surface"]):
                    same = False
                if len(prev_label) > 0 and not (prev_sub_label == evidence["obj_surface"]):
                    same = False
            if not same and args.debug:
                print(sample["masked_sentences"])

        masked_orig_sentence = sample["evidences"][0]["masked_sentence"]
        new_masked_sentence = masked_template.replace("[X]", sub_label).replace("[Y]", MASK)
        lama_examples.append(LamaExample(id,
                                         "T-Rex",
                                         relation,
                                         masked_template,
                                         masked_orig_sentence,
                                         new_masked_sentence,
                                         sub_label,
                                         label.lower()))
    return lama_examples


def read_trex_lpaqa_relation(args, filename, relation, masked_template):
    data = load_file(args, filename)

    lama_examples = []
    for sample in data:
        id = sample["sub_uri"]
        label = sample["obj_label"]
        sub_label = sample["sub_label"]
        new_masked_sentence = masked_template.replace("[X]", sub_label).replace("[Y]", MASK)
        masked_orig_sentence = new_masked_sentence
        lama_examples.append(LamaExample(id,
                                         "T-Rex",
                                         relation,
                                         masked_template,
                                         masked_orig_sentence,
                                         new_masked_sentence,
                                         sub_label,
                                         label.lower()))
    return lama_examples


def get_trex_data(args, lpaqa=False):
    lama_examples = {}
    data_folder = "TREx_train" if lpaqa else "TREx"
    relations, data_path_pre, data_path_post = get_trex_parameters(args,
                                                                   data_folder=data_folder)
    for relation in relations:
        file_path = data_path_pre+relation["relation"]+data_path_post
        if not os.path.exists(file_path):
            continue

        if lpaqa:
            relation_lama_examples = \
                read_trex_lpaqa_relation(args, file_path, relation["label"], relation["template"])
        else:
            relation_lama_examples = \
                read_trex_relation(args, file_path, relation["label"], relation["template"])

        lama_examples[relation["label"]] = (relation_lama_examples, relation["template"])
    return lama_examples


def read_all_data(args, data_path, datasets_file_name):
    if (not os.path.exists(os.path.join(data_path, datasets_file_name))) \
       or args.override_data_dump or args.manual_test:
        # set up data
        trex_data = get_trex_data(args)
        all_data = trex_data
        if not args.manual_test:
            google_re_data = get_google_re_data(args)
            if not(args.debug and len(all_data.values())  >= args.items_for_debug):
                all_data.update(google_re_data)

        pickle.dump(all_data, open(os.path.join(data_path, datasets_file_name), "wb"))
    else:
        print("Reading data from pkl file")
        all_data = pickle.load(open(os.path.join(data_path, datasets_file_name), "rb"))
    return all_data
