import torch
import os
import csv
from datetime import datetime
from tqdm import tqdm, trange

if not torch.cuda.is_available():
    from BERTese.bertese_experiment import preprocess_data
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
    from bertese_experiment import preprocess_data
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


def get_embeddings(model,input, args):
    output = model(input)
    if "bertese" in args.model_type:
        return output[1]
    else:
        return output[1]

if __name__ == "__main__":
    args = cli.parse_args()

    out_dir = "nn_search_" + args.model_type + "_" + str(datetime.now().strftime("%d_%m_%H_%M_%S"))
    if args.debug:
        args = debug_utils.set_debug_args(args)
        out_dir = "debug_" + out_dir
    args.output_dir = os.path.join(args.output_dir, out_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logging_utils.log_args(args, args.output_dir)

    logs_pre_path = args.output_dir
    data_path = args.data_path
    data_path, logs_pre_path = logging_utils.init_log_files(data_path, logs_pre_path)
    filter_met = filter_to_bert_correct_result if args.filter_identity else None

    training_in_sentence_formatter = formatter.identity,
    inference_in_sentance_formatter = formatter.identity,
    data_filter = filter_met,
    log_file_path = logs_pre_path + "/" + args.model_type + "_training_sentence_log.txt",
    result_file_path = logs_pre_path + "/" + args.model_type + "_training_sentence_results.csv",
    data_path = data_path

    model_type = args.model_type

    dev_examples, log_file, lower_model_name, nee, result_file, test_examples, train_examples, upper_model_name = \
        preprocess_data(args, None, data_filter, data_path, log_file_path, result_file_path, [0.8, 0.1, 0.1])

    model_type = args.model_type

    if "bertese" in args.model_type:
        model, tokenizer = bert_predictor.init_model_by_type(args, model_type, upper_model_name, None,
                                                             bertese_lower_model_name=lower_model_name)
    else:
        model, tokenizer = bert_predictor.init_model_by_type(args, model_type, upper_model_name, None)


