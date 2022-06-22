import numpy as np
import random
import nltk
import torch
from tqdm import tqdm
import os
import json
import faiss
from transformers import BertTokenizer, BertForMaskedLM

if not torch.cuda.is_available():
    from BERTese.utils.lama_utils import get_sub_label, LamaExampleEncoder, LamaExampleDecoder
    from BERTese.utils.logging_utils import print_log_to_file
    import BERTese.bert_utils as bert
    import BERTese.mask_prediction as mask_prediction
else:
    from utils.lama_utils import get_sub_label, LamaExampleEncoder
    import mask_prediction
    from utils.logging_utils import print_log_to_file
    import bert_utils as bert


################################
# ------ Accuracy Methods ---- #
################################
def simple_accuracy(preds, out_labels, do_logs=False, log_writer=None):
    n = len(preds)
    correct = 0
    for i in range(n):
        no_pad_label, no_pad_pred = _remove_pads(out_labels[i]), _remove_pads(preds[i])
        no_pad_label = _remove_dummy_cls(no_pad_label)
        if do_logs and log_writer is not None:
            _log_sample(log_writer, no_pad_label, no_pad_pred)
        if np.array_equal(preds[i], out_labels[i]):
            correct += 1
    return correct/n


def single_token_simple_accuracy(preds, out_labels, do_logs=False, log_writer=None):
    n = len(preds)
    correct = 0
    for i in range(n):
        if do_logs and log_writer is not None:
            _log_sample(log_writer, out_labels[i], preds[i])
        if np.array_equal(out_labels[i].lower(), preds[i].lower()):
            correct += 1
    return correct/n


def t5_exact(preds, out_labels, do_logs=False, log_writer=None):
    n = len(preds)
    correct = 0
    for i in range(n):
        no_pad_label, no_pad_pred = _remove_pads(out_labels[i]), _remove_pads(preds[i])
        no_pad_label = _remove_dummy_cls(no_pad_label)
        if do_logs and log_writer is not None:
            _log_sample(log_writer, no_pad_label, no_pad_pred)
        if np.array_equal(_remove_unk(preds[i]), _remove_unk(out_labels[i])):
            correct += 1
    return correct/n


def bleu_accuracy(preds, out_labels, do_logs=False, log_writer=None, input_tokens=None):
    n = len(preds)
    total_bleu = 0
    for i in range(n):
        y_label = _remove_pads(out_labels[i])
        y_pred = _remove_pads(preds[i])
        x = None
        if input_tokens is not None and len(input_tokens) > i:
            x = _remove_pads(input_tokens[i])
        if do_logs and log_writer is not None:
            _log_sample(log_writer,  y_label, y_pred, x)
        total_bleu += nltk.translate.bleu_score.sentence_bleu([y_pred], y_label)

    return total_bleu/n


def after_upper_model_bleu_accuracy(out_embs, out_labels, tokenizer, upper_model, do_logs=False, log_writer=None):
    total_bleu = 0
    with torch.no_grad():
        output = upper_model(inputs_embeds=torch.tensor(out_embs))[0].detach().numpy()
        predicted = np.argmax(output, axis=2)
        preds = [tokenizer.convert_ids_to_tokens(pred) for pred in predicted]
        output = upper_model(input_ids=torch.tensor([tokenizer.encode(label) for label in out_labels]))[0].detach().numpy()
        predicted = np.argmax(output, axis=2)
        labels = [tokenizer.convert_ids_to_tokens(pred) for pred in predicted]
    n = len(preds)
    for i in range(n):
        assumed_length = len(_remove_pads(out_labels[i]).split())
        y_label = ' '.join(labels[i][1:assumed_length - 1])  # we remove [CLS] and [SEP] predictions
        y_pred = ' '.join(preds[i][1:assumed_length - 1])
        if do_logs and log_writer is not None:
            _log_sample(log_writer,  y_label, y_pred)
        total_bleu += nltk.translate.bleu_score.sentence_bleu([y_pred], y_label)

    return total_bleu/n


def topk_simple_accuracy(topk_preds, out_labels, do_logs=False, log_writer=None):
    n = len(topk_preds)
    correct = 0
    for i in range(n):
        if do_logs and log_writer is not None:
            _log_sample(log_writer, topk_preds, out_labels)
        if out_labels[i] in topk_preds[i]:
            correct += 1
    return correct/n


##################################
# ------ Data Split Methods ---- #
##################################
def split_test_dev_train(data, ratio):
    train_size, dev_size, test_size = ratio
    test = {}
    dev = {}
    train = {}

    for relation in data:
        examples, relation_template = data[relation]
        valid_for_train_examples = [e for e in examples if e.valid_for_train]
        valid_for_test_examples = [e for e in examples if not e.valid_for_train]
        random.shuffle(valid_for_train_examples)
        random.shuffle(valid_for_test_examples)

        train_examples = valid_for_train_examples[:int((len(valid_for_train_examples)) * train_size)]
        test_examples = valid_for_train_examples[int(len(valid_for_train_examples) * train_size) + 1:int(len(examples) * (train_size + test_size))]
        dev_examples = valid_for_train_examples[int(len(valid_for_train_examples) * (train_size + test_size)) + 1:]

        test_examples = test_examples + valid_for_test_examples[:int((len(valid_for_test_examples)) * 0.5)]
        dev_examples = dev_examples + valid_for_test_examples[int((len(valid_for_test_examples)) * 0.5)+1:]

        train[relation] = (train_examples, relation_template)
        dev[relation] = (dev_examples, relation_template)
        test[relation] = (test_examples, relation_template)

    return test, dev, train


def split_relations_test_dev_train(data, ratio):
    train_size, dev_size, test_size = ratio
    relations = [k for k in data.keys()]
    random.shuffle(relations)

    train_keys = relations[:int((len(relations)) * train_size)]
    test_keys = relations[int(len(relations) * train_size) + 1:int(len(relations) * (train_size + test_size))]
    dev_keys = relations[int(len(relations) * (train_size + test_size)) + 1:]

    train = {key: value for key, value in data.items() if key in train_keys}
    #valid_train = {}
    #for key in train:
    #    valid_train[key] = ([e for e in train[key][0] and e.valid_for_train], train[key][1])
    test = {key: value for key, value in data.items() if key in test_keys}
    dev = {key: value for key, value in data.items() if key in dev_keys}

    return test, dev, train


###################################
# ------ Data Filter Methods ---- #
###################################
def no_filter(data, model, tokenizer, cased=False):
    return data


def filter_to_bert_correct_result(data, model, tokenizer, cased=False):
    results = {}
    data_relations = list(data.keys())
    for r in tqdm(range(len(data_relations)), desc="Filter Relation Data"):
        relation = data_relations[r]
        filtered_examples = []
        examples, relation_template = data[relation]
        for i in range(len(examples)):
            example = examples[i]

            example.label = example.label if cased else example.label.lower()
            example.sub_label = example.sub_label if cased else example.sub_label.lower()
            subject = get_sub_label(example.snippet, example.sub_label)
            if subject is None:
                continue
            template_masked_sentence = example.new_masked_sentence
            if template_masked_sentence.count(bert.MASK) > 1:
                continue
            if template_masked_sentence.count(bert.MASK) < 1:
                continue

            try:
                predicted_template_masks = mask_prediction.predict_mask_token(model=model,
                                                                              tokenizer=tokenizer,
                                                                              sentence_a=template_masked_sentence,
                                                                              pred_count=10,
                                                                              cased=cased)
                if predicted_template_masks[0] == example.label:
                    filtered_examples.append(example)
            except RuntimeError as e:
                continue

        results[relation] = (filtered_examples, relation_template)
    return results


###################################
# ------ Misc Methods ---- #
###################################
def split_and_filter(all_data, args, cache_dir, data_filter,
                     dev_file_path, model_name, split_ratio,
                     test_fila_path, train_file_path, split_method = split_test_dev_train):
    if args.override_data_split or \
            not (os.path.exists(test_fila_path) and
                 os.path.exists(train_file_path) and
                 os.path.exists(dev_file_path)):
        test_data, dev_data, train_data = split_method(all_data, split_ratio)

        if data_filter is not None:
            vanilla_model, vanilla_tokenizer = mask_prediction.init_vanilla_mask_prediction_model(
                args.mask_predict_model_name, cache_dir)

            test_data = data_filter(test_data, vanilla_model, vanilla_tokenizer, cased="cased" in model_name)
            train_data = data_filter(train_data, vanilla_model, vanilla_tokenizer, cased="cased" in model_name)
            dev_data = data_filter(dev_data, vanilla_model, vanilla_tokenizer, cased="cased" in model_name)

        json.dump(test_data, open(test_fila_path, 'w'), indent=4, cls=LamaExampleEncoder)
        json.dump(train_data, open(train_file_path, 'w'), indent=4, cls=LamaExampleEncoder)
        json.dump(dev_data, open(dev_file_path, 'w'), indent=4, cls=LamaExampleEncoder)
    else:
        test_data = json.load(open(test_fila_path, 'r'))#, cls=LamaExampleDecoder)
        train_data = json.load(open(train_file_path, 'r'))#, cls=LamaExampleDecoder)
        dev_data = json.load(open(dev_file_path, 'r'))#, cls=LamaExampleDecoder)
    return train_data, dev_data, test_data


def filter(test_data, dev_data, train_data, args, cache_dir, data_filter, model_name, dev_file_path,
                     test_file_path, train_file_path):
    if args.override_data_dump or args.debug or \
            not (os.path.exists(test_file_path) and
                 os.path.exists(train_file_path) and
                 os.path.exists(dev_file_path)):

        if data_filter is not None:
            vanilla_model, vanilla_tokenizer = mask_prediction.init_vanilla_mask_prediction_model(
                args.mask_predict_model_name, cache_dir)

            test_data = data_filter(test_data, vanilla_model, vanilla_tokenizer, cased="cased" in model_name)
            train_data = data_filter(train_data, vanilla_model, vanilla_tokenizer, cased="cased" in model_name)
            dev_data = data_filter(dev_data, vanilla_model, vanilla_tokenizer, cased="cased" in model_name)

        json.dump(test_data, open(test_file_path, 'w'), indent=4, cls=LamaExampleEncoder)
        json.dump(train_data, open(train_file_path, 'w'), indent=4, cls=LamaExampleEncoder)
        json.dump(dev_data, open(dev_file_path, 'w'), indent=4, cls=LamaExampleEncoder)
    else:
        test_data = json.load(open(test_file_path, 'r'))#, cls=LamaExampleDecoder)
        train_data = json.load(open(train_file_path, 'r'))#, cls=LamaExampleDecoder)
        dev_data = json.load(open(dev_file_path, 'r'))#, cls=LamaExampleDecoder)

    print("filed saved to:", train_file_path)
    return train_data, dev_data, test_data




def save_model(args, global_step, model, optimizer, scheduler, tokenizer, logger):
    if args.dont_save_model:
        return
    else:
        # Save model checkpoint
        checkpoint_prefix = "checkpoint"
        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger("Saving model checkpoint to %s" % output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        logger("Saving optimizer and scheduler states to %s" % output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger("finished saving...")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    return device, n_gpu


def _log_sample(log_writer, y, y_pred, x=None):
    if random.randint(0, 100) % 10 == 0:
        if x is not None:
            print_log_to_file(log_writer, "\nX: {}\nY: {}\nPRED_Y: {}\n".format(x, y, y_pred))
        else:
            print_log_to_file(log_writer, "\nY: {}\nPRED_Y: {}\n".format(y, y_pred))


def _remove_dummy_cls(y):
    return y[len(bert.CLS)+1:]


def _remove_unk(values):
    while values.strip().strip('⁇') != values:
        values = values.strip().strip('⁇')
    return values


def _remove_pads(value):
    value_end_char = min(value.index("[PAD]") if "[PAD]" in value else len(value), value.index("nivorousnivorous") if "nivorousnivorous" in value else len(value))
    value_no_pad = value[:value_end_char]
    return value_no_pad


def get_attention_mask(tensor):
    attention_mask = tensor.clone()
    attention_mask[attention_mask > 0] = 1
    return attention_mask


def get_nearset_token(preds, nbrs):
    # dists, neighbours = nbrs.kneighbors(preds.reshape(-1, preds.shape[2]))
    # neighbours = neighbours.reshape(preds.shape[0], preds.shape[1])
    index = faiss.IndexFlatL2(nbrs.shape[-1])
    index.add(nbrs)
    D, I = index.search(preds.reshape(-1, preds.shape[2]), 1)
    return I.reshape(preds.shape[0], -1)
