import torch
import os
import csv
from datetime import datetime
from tqdm import tqdm, trange

if not torch.cuda.is_available():
    from BERTese.utils.lama_utils import to_examples_list
    import BERTese.snippet_formatter as formatter
    import BERTese.training_utils as training_utils
    import BERTese.readers as readers
    import BERTese.bert_predictor as bert_predictor
    import BERTese.cli as cli
    import BERTese.bertese_training as training
    import BERTese.utils.debug_utils as debug_utils
    import BERTese.utils.logging_utils as logging_utils
    import BERTese.nlp_utils as nlp_utils
    from BERTese.training_utils import filter_to_bert_correct_result
    from BERTese.mask_prediction import predict_mask_token as vanilla_bert_mask_prediction
    from BERTese.bert_predictor import init_vanilla_mask_prediction_model
    import BERTese.mask_prediction as mask_prediction
else:
    from utils.lama_utils import to_examples_list
    import snippet_formatter as formatter
    import training_utils
    import readers
    import bert_predictor
    import cli
    import bertese_training as training
    import utils.debug_utils as debug_utils
    import utils.logging_utils as logging_utils
    import nlp_utils as nlp_utils
    from training_utils import filter_to_bert_correct_result
    from bert_predictor import init_vanilla_mask_prediction_model
    import mask_prediction as mask_prediction


def lama_bertese_experiment(args, data_filter, model_type, cache_dir=None,
                            training_in_sentence_formatter=formatter.identity,
                            inference_in_sentance_formatter=formatter.identity,
                            log_file_path="/results/log.txt", result_file_path="/results/final_results.csv",
                            data_path="/data/",
                            split_ratio=[0.8, 0.1, 0.1]):

    dev_examples, log_file, lower_model_name, nee, result_file, test_examples, train_examples, upper_model_name = \
        preprocess_data(args, cache_dir, data_filter, data_path, log_file_path, result_file_path, split_ratio)

    # Train and return model
    logging_utils.print_log_to_screen("RUNNING WITH MODEL TYPE:" + model_type)
    model, tokenizer = bert_predictor.init_model_by_type(args, model_type, upper_model_name, cache_dir,
                                                         bertese_lower_model_name=lower_model_name)

    if args.do_train:
        vanilla_bert_model = None
        if args.bert_pred_sanity_test:
            vanilla_bert_model, _ = init_vanilla_mask_prediction_model(args.upper_model_name)

        model, train_out, dev_out, test_out, = \
            training.train_and_eval_lama(args, model, train_examples, dev_examples, test_examples,
                                         training_in_sentence_formatter, inference_in_sentance_formatter, nee,
                                         tokenizer,vanilla_bert_model)

        result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row = ["all", "all", len(train_examples), "all", train_out["acc"]]

        if "top10_acc" in train_out:
            row += [train_out["top10_acc"]]
        else:
            row += ["NA"]

        if args.do_eval_dev:
            row += [len(dev_examples), dev_out["acc"]]

            if "top10_acc" in dev_out:
                row += [dev_out["top10_acc"]]
            else:
                row += ["NA"]

        if args.do_eval_test:
            row += [len(test_examples), test_out["eval_acc"]]
            if "top10_acc" in dev_out:
                row += [test_out["top10_acc"]]
            else:
                row += ["NA"]

        result_writer.writerow(row)
    elif args.do_eval_dev:
        print("need to implement eval only mode")

    result_file.flush()
    result_file.close()

    log_file.flush()
    log_file.close()


def preprocess_data(args, cache_dir, data_filter, data_path, log_file_path, result_file_path, split_ratio):
    upper_model_name = args.upper_model_name
    lower_model_name = args.model_name
    datasets_file_name = "training_data.pkl"
    nee_file_name = "nee.pkl"
    if args.lpaqa:
        data_path = os.path.join(data_path, "TREx_LPAQA_split")
    else:
        data_path = os.path.join(data_path, args.split_method)
    if data_filter is not None:
        data_path = os.path.join(data_path, "_" + data_filter.__name__ + "_" + upper_model_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    test_file_path = os.path.join(data_path, "test.json")
    train_file_path = os.path.join(data_path, "train.json")
    dev_file_path = os.path.join(data_path, "dev.json")
    result_file = open(result_file_path, "w+", 1)
    log_file = open(log_file_path, "w+", 1)
    result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    columns = ["Dataset", "Relation", "Train Sample Size", "Template", "Train P@1", "Train P@10"]
    if args.do_eval_dev:
        columns += ["Dev Sample Size", "Dev P@1", "Dev P@10"]
    if args.do_eval_test:
        columns += ["Test P@1", "Test Sample Size", "Test P@10"]
    result_writer.writerow(columns)
    # Read data
    all_data = readers.read_all_data(args, data_path, datasets_file_name)
    if args.lpaqa:
        train_data = readers.get_trex_data(args, lpaqa=True)
        dev_data = readers.get_trex_data(args, lpaqa=False)
        test_data = dev_data

        train_data, dev_data, test_data = training_utils.filter(test_data, dev_data, train_data,
                                                                args, cache_dir, data_filter, upper_model_name,
                                                                dev_file_path,
                                                                test_file_path, train_file_path)

    else:
        logging_utils.print_log_to_screen("split and filter data")
        if args.split_method == "random_split":
            split_method = training_utils.split_test_dev_train
        elif args.split_method == "relation_split":
            split_method = training_utils.split_relations_test_dev_train
        else:
            raise ValueError("split method can only be: random_split or relation_split")

        train_data, dev_data, test_data = training_utils.split_and_filter(all_data, args, cache_dir, data_filter,
                                                                          dev_file_path, upper_model_name, split_ratio,
                                                                          test_file_path, train_file_path, split_method)
    logging_utils.print_log_to_screen("finished split and filter data!")
    # Read NEE
    nee = nlp_utils.extract_entities(all_data, data_path, nee_file_name)

    train_examples = to_examples_list(train_data)
    dev_examples = to_examples_list(dev_data)
    test_examples = to_examples_list(test_data)

    return dev_examples, log_file, lower_model_name, nee, result_file, test_examples, train_examples, upper_model_name


def lama_bert_experiment(args, data_filter, model_type, cache_dir=None,
                         in_sentence_formatter=formatter.identity,
                         log_file_path="/results/log.txt", result_file_path="/results/final_results.csv",
                         data_path="/data/",
                         split_ratio=[0.8, 0.1, 0.1]):

    model_name = args.upper_model_name
    datasets_file_name = "training_data.pkl"
    nee_file_name = "nee.pkl"

    if args.lpaqa:
        data_path = os.path.join(data_path, "TREx_LPAQA_split")
    else:
        data_path = os.path.join(data_path, args.split_method)

    if data_filter is not None:
        data_path = os.path.join(data_path, "_" + data_filter.__name__ +"_"+model_name)

    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    test_file_path = os.path.join(data_path, "test.json")
    train_file_path = os.path.join(data_path, "train.json")
    dev_file_path = os.path.join(data_path, "dev.json")
    result_file = open(result_file_path, "w+", 1)
    log_file = open(log_file_path, "w+", 1)

    result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    columns = ["Dataset", "Relation", "Train Sample Size", "Template", "Train P@1", "Train P@10"]

    if args.do_eval_dev:
        columns += ["Dev Sample Size", "Dev P@1", "Dev P@10"]

    if args.do_eval_test:
        columns += ["Test P@1", "Test Sample Size", "Test P@10"]

    result_writer.writerow(columns)

    # Read data
    if args.lpaqa:
        train_data = readers.get_trex_data(args, lpaqa=True)
        dev_data = readers.get_trex_data(args, lpaqa=False)
        test_data = dev_data

        train_data, dev_data, test_data = training_utils.filter(test_data, dev_data, train_data,
                                                                args, cache_dir, data_filter,model_name,
                                                                dev_file_path,
                                                                test_file_path, train_file_path)
    else:
        all_data = readers.read_all_data(args, data_path, datasets_file_name)
        logging_utils.print_log_to_screen("split and filter data")
        if args.split_method == "random_split":
            split_method = training_utils.split_test_dev_train
        elif args.split_method == "relation_split":
            split_method = training_utils.split_relations_test_dev_train
        else:
            raise ValueError("split method can only be: random_split or relation_split")

        train_data, dev_data, test_data = training_utils.split_and_filter(all_data, args, cache_dir, data_filter,
                                                                          dev_file_path, model_name, split_ratio,
                                                                          test_file_path, train_file_path, split_method)

    logging_utils.print_log_to_screen("finished split and filter data!")

    # Read NEE
    nee = nlp_utils.extract_entities(train_data, data_path, nee_file_name)

    # Train and return model
    logging_utils.print_log_to_screen("RUNNING WITH MODEL TYPE:" + model_type)
    pre_trained_bert_model, tokenizer = bert_predictor.init_model_by_type(args, model_type, model_name, cache_dir)
    logging_utils.print_log_to_screen("finished loading the model")

    train_examples = to_examples_list(train_data)
    dev_examples = to_examples_list(dev_data)
    test_examples = to_examples_list(test_data)
    if args.do_train:
        model, train_out, dev_out, test_out = training.train_and_eval_lama(args, pre_trained_bert_model, train_examples,
                                                                           dev_examples, test_examples,
                                                                           in_sentence_formatter,
                                                                           formatter.identity,
                                                                           nee, tokenizer)
        train_acc = train_out["acc"]
        train_top10_acc = train_out["top10_acc"] if "top10_acc" in train_out else "NA"

        if args.do_eval_dec:
            dev_acc = dev_out["acc"]
            dev_top10_acc = dev_out["top10_acc"] if "top10_acc" in dev_out else "NA"

        if args.do_eval_dec:
            test_acc = test_out["acc"]
            test_top10_acc = test_out["top10_acc"] if "top10_acc" in test_out else "NA"

    else:
        model,_ = init_vanilla_mask_prediction_model(model_name)
        train_acc, train_top10_acc = evaluate_bert(train_examples, model, tokenizer)

        if args.do_eval_dev:
            dev_acc, dev_top10_acc = evaluate_bert(dev_examples, model, tokenizer)

        if args.do_eval_test:
            test_acc, test_top10_acc = evaluate_bert(test_examples, model, tokenizer)

    result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    row = ["all", "all", len(train_examples), "all", train_acc, train_top10_acc]

    if args.do_eval_dev:
        row += [len(dev_examples), dev_acc, dev_top10_acc]

    if args.do_eval_test:
        row += [len(test_examples), test_acc, test_top10_acc]

    print(columns)
    print(row)
    result_writer.writerow(row)

    result_file.flush()
    result_file.close()
    log_file.flush()
    log_file.close()


def evaluate_bert(dev_examples, model, tokenizer):
    dev_acc = 0.0
    dev_top10_acc = 0.0
    for e in tqdm(dev_examples, desc="Evaluating"):
        bert_pred = bert_predictor.mask_prediction.predict_mask_token(model=model, tokenizer=tokenizer,
                                                                      sentence_a=e.masked_sentence)
        if e.label in bert_pred:
            dev_top10_acc += 1
            if e.label == bert_pred[0]:
                dev_acc += 1
    return dev_acc/len(dev_examples), dev_top10_acc/len(dev_examples)


def _run_bert_exp(args):
    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)
    filter_met = filter_to_bert_correct_result if args.filter_identity else None
    logging_utils.log_args(args, logs_pre_path)

    lama_bert_experiment(args,
                            in_sentence_formatter=formatter.identity,
                            data_filter=filter_met,
                            log_file_path=logs_pre_path + "/"+args.model_type+"_sentence_log.txt",
                            result_file_path=logs_pre_path + "/"+args.model_type+"_sentence_results.csv",
                            data_path=data_path,
                            model_type=args.model_type)


def _run_bertese_exp(args):
    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)
    filter_met = filter_to_bert_correct_result if args.filter_identity else None

    lama_bertese_experiment(args,
                            training_in_sentence_formatter=formatter.identity,
                            inference_in_sentance_formatter=formatter.identity,
                            data_filter=filter_met,
                            log_file_path=logs_pre_path + "/"+args.model_type+"_training_sentence_log.txt",
                            result_file_path=logs_pre_path + "/"+args.model_type+"_training_sentence_results.csv",
                            data_path=data_path,
                            model_type=args.model_type)


def _run_bertese_exp(args, train_formatter):
    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)
    filter_met = filter_to_bert_correct_result if args.filter_identity else None

    lama_bertese_experiment(args,
                            training_in_sentence_formatter=train_formatter,
                            inference_in_sentance_formatter=formatter.identity,
                            data_filter=filter_met,
                            log_file_path=logs_pre_path + "/" + args.model_type + "_training_sentence_log.txt",
                            result_file_path=logs_pre_path + "/" + args.model_type + "_training_sentence_results.csv",
                            data_path=data_path,
                            model_type=args.model_type)


if __name__ == "__main__":
    args = cli.parse_args()

    out_dir = args.model_type + "_" + str(datetime.now().strftime("%d_%m_%H_%M_%S"))
    if args.debug:
        args = debug_utils.set_debug_args(args)
        out_dir = "debug_" + out_dir
    args.output_dir = os.path.join(args.output_dir, out_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logging_utils.log_args(args, args.output_dir)

    if "bertese" in args.model_type:
        if "rewrite" in args.model_type:
            _run_bertese_exp(args, formatter.rewrite_identity)
        else:
            _run_bertese_exp(args, formatter.identity)
    elif "lama_bert" in args.model_type: #This is the Baseline
        _run_bert_exp(args)
    else:
        raise ValueError("Invalid Exp Type")
