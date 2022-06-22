import torch
import os

if not torch.cuda.is_available():
    from BERTese.utils.lama_utils import to_examples_list
    from BERTese.utils.logging_utils import print_log_to_screen as print_log
    import BERTese.snippet_formatter as formatter
    import BERTese.training_utils as training_utils
    import BERTese.readers as readers
    import BERTese.bert_predictor as bert_predictor
    import BERTese.cli as cli
    import BERTese.seq2seq_training as training
    import BERTese.utils.debug_utils as debug_utils
    import BERTese.utils.logging_utils as logging_utils
    import BERTese.nlp_utils as nlp_utils
else:
    from utils.lama_utils import to_examples_list
    from utils.logging_utils import print_log_to_screen as print_log
    import snippet_formatter as formatter
    import training_utils
    import readers
    import bert_predictor
    import cli
    import seq2seq_training as training
    import utils.debug_utils as debug_utils
    import utils.logging_utils as logging_utils
    import nlp_utils as nlp_utils


def lama_seq2seq_experiment(args, data_filter, model_type, cache_dir=None,
                            in_sentence_formatter=formatter.identity,
                            out_sentence_formatter=formatter.identity,
                            log_file_path="/results/log.txt", result_file_path="/results/final_results.csv",
                            data_path="/data/",
                            split_ratio=[0.8, 0.1, 0.1]):

    model_name = args.model_name
    datasets_file_name = "training_data.pkl"
    nee_file_name = "nee.pkl"

    if args.lpaqa:
        data_path = os.path.join(data_path, "TREx_LPAQA_split")
    else:
        data_path = os.path.join(data_path, args.split_method)

    test_fila_path = os.path.join(data_path, "test.json")
    train_file_path = os.path.join(data_path, "train.json")
    dev_file_path = os.path.join(data_path, "dev.json")
    log_file = open(log_file_path, "w+", 1)

    # Read data
    all_data = readers.read_all_data(args, data_path, datasets_file_name)
    if args.lpaqa:
        train_data = readers.get_trex_data(args, lpaqa=True)
        dev_data = readers.get_trex_data(args, lpaqa=False)
        test_data = dev_data
    else:
        train_data, dev_data, test_data = training_utils.split_and_filter(all_data, args, cache_dir, data_filter,
                                                                          dev_file_path,model_name, split_ratio,
                                                                          test_fila_path, train_file_path)

    # Read NEE
    nee = nlp_utils.extract_entities(all_data, data_path, nee_file_name, args.override_nee)

    # Train and return model
    print_log("RUNNING WITH MODEL TYPE:" + model_type)
    model, tokenizer = bert_predictor.init_model_by_type(args, model_type, model_name, cache_dir)

    train_examples = to_examples_list(train_data)
    dev_examples = to_examples_list(dev_data)
    test_examples = to_examples_list(test_data)
    print_log("Training set size: {}\nDev set size: {}\nTest set size: {}".
              format(len(train_examples), len(dev_examples), len(test_examples)))

    if args.do_train:
        training.train_and_eval(args, model, train_examples, dev_examples, test_examples,
                                in_sentence_formatter, out_sentence_formatter, nee, tokenizer, tokenizer.vocab_size)

    log_file.flush()
    log_file.close()


def _run_identity_exp(args):
    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)
    lama_seq2seq_experiment(args,
                            in_sentence_formatter=formatter.identity,
                            out_sentence_formatter=formatter.identity,
                            data_filter=None,
                            log_file_path=logs_pre_path + "/identity_training_sentence_log.txt",
                            result_file_path=logs_pre_path + "/identity_training_sentence_results.csv",
                            data_path=data_path,
                            model_type=args.model_type)


def _run_rewrite_identity_exp(args):
    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)
    print("logging steps:", args.logging_steps)
    lama_seq2seq_experiment(args,
                            in_sentence_formatter=formatter.identity,
                            out_sentence_formatter=formatter.rewrite_identity,
                            data_filter=None,
                            log_file_path=logs_pre_path + "/"+args.model_type+"_training_sentence_log.txt",
                            result_file_path=logs_pre_path + "/"+args.model_type+"_training_sentence_results.csv",
                            data_path=data_path,
                            model_type=args.model_type)


def _run_rewrite_identity_with_entity_exp(args):
    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)

    #if True or "constant" in args.subject_replace:
    in_formatter = formatter.replace_subject_with_constant
    out_formatter = formatter.rewrite_replace_subject_with_constant
    if "relation" in args.subject_replace:
        in_formatter = formatter.replace_subject_with_relation
        out_formatter = formatter.rewrite_replace_subject_with_relation()

    lama_seq2seq_experiment(args,
                            in_sentence_formatter=in_formatter,
                            out_sentence_formatter=out_formatter,
                            data_filter=None,
                            log_file_path=logs_pre_path + "/"+args.model_type+"_training_sentence_log.txt",
                            result_file_path=logs_pre_path + "/"+args.model_type+"_training_sentence_results.csv",
                            data_path=data_path,
                            model_type=args.model_type)


if __name__ == "__main__":
    args = cli.parse_args()
    if args.debug:
        args = debug_utils.set_debug_args(args)

    if args.subject_replace != "none":  # The subject is the entity
        _run_rewrite_identity_with_entity_exp(args)
    if "identity" in args.model_type:
        _run_identity_exp(args)
    if "rewrite" in args.model_type:
        _run_rewrite_identity_exp(args)
    else:
        "INVALID EXP TYPE"
