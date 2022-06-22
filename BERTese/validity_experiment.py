import torch
from tqdm import tqdm
import csv
import os
import sys
import pickle

if not torch.cuda.is_available():
    from BERTese.utils.lama_utils import get_sub_label
    from BERTese.utils.logging_utils import print_log_to_screen as print_log
    from BERTese.snippet_formatter import NEE
    import BERTese.snippet_formatter as formatter
    import BERTese.readers as readers
    import BERTese.bert_predictor as bert_predictor
    import BERTese.cli as cli
else:
    from lama_utils import get_sub_label
    from logging_utils import print_log_to_screen as print_log
    from snippet_formatter import NEE
    import snippet_formatter as formatter
    import readers
    import bert_predictor
    import cli


OVERRIDE_DEBUG = False
SUBJECT_DEBUG = False
MORE_THAN_ONE_MASK_DEBUG = False
NO_MASK_DEBUG = True
BERT_EXCEPTION_DEBUG = False


def lama_supervised_experiment(args, datasets, snippet_formatter, nee=None, model_name="lower_bert-large-uncased", cache_dir=None, only_alphanumeric=False,
                               log_file_path="/results/log.txt", result_file_path="/results/final_results.csv"):
    model, tokenizer = bert_predictor.init_vanilla_mask_prediction_model(model_name, cache_dir)

    result_file = open(result_file_path, "w+", 1)
    log_file = open(log_file_path, "w+", 1)

    result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    result_writer.writerow(["Dataset", "Relation", "Sample Size", "Valid Sample Size", "Template",
                            "Template P@1", "Template P@10", "Template MRR",
                            "Snippet ReWrite P@1", "Snippet ReWrite P@10", "Snippet ReWrite MRR",
                            "Snippet ReWrite Better@10", "Template Better@10"])

    for ds_name, ds in datasets:
        run_dataset_analysis(args, model, tokenizer, ds_name, ds,
                             log_file, model_name, only_alphanumeric, snippet_formatter, nee,
                             result_writer, cache_dir=cache_dir)
        result_file.flush()
        log_file.flush()

    result_file.close()
    log_file.close()


def run_dataset_analysis(args, model, tokenizer, dataset, data, log_file, model_name, only_alphanumeric,
                         sentence_formatter, nee, result_writer, cased=False, cache_dir = None):
    total_more_than_one_mask = 0
    total_no_mask_value = 0
    total_not_logged_examples = 0
    total_skip_sub_label_missing = 0
    total_bert_exception = 0
    total_skip_long = 0
    total_skip_short = 0
    stop = False

    log_file.write("log for {}\n".format(dataset))
    for relation in data:
        if stop:
            break

        log_file.write("Relation: {}, {}\n".format(dataset, relation))
        print_log("Relation:{}, {}\n".format(dataset, relation))
        more_than_one_mask = 0
        no_mask_value = 0
        not_logged_examples = 0
        skip_sub_label_missing = 0
        bert_exception = 0
        skip_long = 0
        skip_short = 0
        not_valid_for_train = 0

        # stats
        template_better = 0
        snippet_rewrite_better = 0

        # Mean reciprocal rank
        template_mrr = 0.0
        snippet_rewrite_mrr = 0.0

        # Precision at (default 10)
        snippet_rewrite_precision = 0.0
        snippet_rewrite_precision1 = 0.0

        # Precision at (default 10)
        template_precision = 0.0
        template_precision1 = 0.0

        examples, relation_template = data[relation]
        checked_examples = len(examples)
        for i in tqdm(range(len(examples))):
            is_logged = True
            example = examples[i]

            if len(args.debug_uuid) > 0:
                if not (example.uuid == args.debug_uuid):
                    continue
                else:
                    stop = True

            example.label = example.label if cased else example.label.lower()
            example.sub_label = example.sub_label if cased else example.sub_label.lower()

            subject = get_sub_label(example.snippet, example.sub_label)
            if subject is None:
                if args.verbose or SUBJECT_DEBUG:
                    print_log("SUBJECT NOT FOUND IN SNIPPET! SKIPPED EXAMPLE\n "
                              "RELATION:{}\n UUID:{}\n SUBJECT:{}\n label:{}\n SNIPPET:{}\n"
                              .format(relation, example.uuid, example.sub_label, example.label, example.snippet))
                skip_sub_label_missing += 1
                checked_examples -= 1
                continue

            # ----------------------------------
            # -- Run snippet masked sentence --
            # ----------------------------------
            try:
                snippet_masked_sentence, valid_for_train = sentence_formatter(example, nee)
                example.valid_for_train = valid_for_train
            except Exception as e:
                checked_examples -= 1
                print_log(e)
                print_log(example)
                continue

            try:
                log_file.write("{}\n".format(example))
                log_file.flush()
            except UnicodeEncodeError as e:
                log_file.write("uuid:{}\n LOGGING ERROR\n ".format(example.uuid))
                log_file.flush()
                is_logged = False

            try:
                log_file.write(" Snippet Masked sentence: {}\n".format(snippet_masked_sentence))
            except UnicodeEncodeError as e:
                is_logged = False

            if not valid_for_train:
                checked_examples -= 1
                log_file.write(" ----- Example is not valid for training ---- \n\n")
                not_valid_for_train +=1
                continue

            if len(snippet_masked_sentence.split()) > 100:
                checked_examples -= 1
                skip_long += 1
                continue

            if len(snippet_masked_sentence.split()) < 1:
                print_log(snippet_masked_sentence.split())
                checked_examples -= 1
                skip_short += 1
                continue

            if snippet_masked_sentence.count(bert_predictor.MASK) > 1:
                more_than_one_mask += 1
                if args.verbose or MORE_THAN_ONE_MASK_DEBUG:
                    print_log("ERROR! MORE THAN ONE MASK IN FORMATTED SNIPPET (Skipped): {}\n{}\n{}\n".format(dataset, relation, example.uuid, snippet_masked_sentence),
                              "ERROR! MORE THAN ONE MASK IN FORMATTED SNIPPET (Skipped): {}\n{}\n".format(dataset, relation, example.uuid))
                checked_examples -= 1
                continue

            if snippet_masked_sentence.count(bert_predictor.MASK) < 1:
                no_mask_value += 1
                if args.verbose or NO_MASK_DEBUG:
                    print_log("\nERROR! NO MASK VALUE IN FORMATTED SNIPPET:{}\n UUID:{}\n RELATION:{} \n "
                              "formatted snippet masked input:{}\n".format(dataset, example.uuid, relation, snippet_masked_sentence),
                              "\nERROR! NO MASK VALUE IN FORMATTED SNIPPET:{}\n UUID:{}\n RELATION:{} \n "
                              "formatted snippet masked input:{}\n".format(dataset, example.uuid, relation, "unicode error"))
                    print_log(example)
                checked_examples -= 1
                continue
            try:
                predicted_masks_snippet_rewrite = bert_predictor.bert_predict_mask_token(model=model,
                                                                                         tokenizer=tokenizer,
                                                                                         model_name=model_name,
                                                                                         sentence_a=snippet_masked_sentence,
                                                                                         pred_count=10,
                                                                                         only_alphanumeric=only_alphanumeric)
            except:
                e = sys.exc_info()[0]
                if args.verbose or args.verbose_e or BERT_EXCEPTION_DEBUG:
                    print_log("Bert exception! evidence snippet \n Input len:{}\n Input:{}\n".format(len(snippet_masked_sentence), snippet_masked_sentence),
                              "Bert exception! UNICODE ERROR")
                    print(e)
                print(example)
                if args.stop_on_e:
                    stop = True
                    break
                checked_examples -= 1
                bert_exception += 1
                continue

            # ----------------------------------
            # -- Run template masked sentence --
            # ----------------------------------
            template_masked_sentence = example.new_masked_sentence

            if template_masked_sentence.count(bert_predictor.MASK) > 1:
                checked_examples -= 1
                more_than_one_mask += 1
                if args.verbose or MORE_THAN_ONE_MASK_DEBUG:
                    print_log("ERROR! MORE THAN ONE MASK VALUE IN NEW TEMPLATE: {}\n{}\n{}\n"
                              .format(dataset, example.uuid, template_masked_sentence),
                              "ERROR! MORE THAN ONE MASK VALUE IN TEMPLATE: {}\n{}\n"
                              .format(dataset, example.uuid))
                continue
            if template_masked_sentence.count(bert_predictor.MASK) < 1:
                checked_examples -= 1
                no_mask_value += 1
                if args.verbose or NO_MASK_DEBUG:
                    print_log("ERROR! NO MASK VALUE IN NEW TEMPLATE: {}\n{}\n{}\n"
                              .format(dataset, example.uuid, template_masked_sentence),
                              "ERROR! NO MASK VALUE IN NEW TEMPLATE: {}\n{}\n"
                              .format(dataset, example.uuid))

                continue

            try:
                predicted_template_masks = bert_predictor.bert_predict_mask_token(model=model,
                                                                                  tokenizer=tokenizer,
                                                                                  model_name=model_name,
                                                                                  sentence_a=template_masked_sentence,
                                                                                  pred_count=10,
                                                                                  only_alphanumeric=only_alphanumeric,
                                                                                  cache_dir = cache_dir)
            except RuntimeError as e:
                if args.verbose_e or args.verbose or BERT_EXCEPTION_DEBUG:
                    print_log(
                        "Bert exception for template!\n  RELATION:{}\n UUID:{}\n Input len:{}\n Input:{}\n".format(
                            relation, example.uuid, len(template_masked_sentence), template_masked_sentence),
                        "Bert exception! UNICODE ERROR")
                    print(e)
                if args.stop_on_e:
                    stop = True
                    break
                checked_examples -= 1
                bert_exception += 1
                continue

            try:
                log_file.write("predictions for template \"{}\":\n {}\n".format(template_masked_sentence,
                                                                                predicted_template_masks))
            except UnicodeEncodeError as e:
                is_logged = False

            try:
                log_file.write("predictions snippet rewrite \"{}\":\n {}\n".format(snippet_masked_sentence,
                                                                                   predicted_masks_snippet_rewrite))
            except UnicodeEncodeError as e:
                is_logged = False

            # Compute Precision\MRR
            if example.label.lower() in predicted_template_masks:
                template_mask_rank = predicted_template_masks.index(example.label.lower()) + 1
                if template_mask_rank == 1:
                    template_precision1 += 1
                template_precision += 1
                template_mrr += (1 / template_mask_rank)

            if example.label.lower() in predicted_masks_snippet_rewrite:
                snippet_rewrite_mask_rank = predicted_masks_snippet_rewrite.index(example.label.lower()) + 1
                if snippet_rewrite_mask_rank == 1:
                    snippet_rewrite_precision1 += 1
                snippet_rewrite_precision += 1
                snippet_rewrite_mrr += (1 / snippet_rewrite_mask_rank)

            log_file.write("correct prediction for template:{}\n".format(example.label.lower() in predicted_template_masks))
            log_file.write("correct prediction after snippet rewrite:{}\n".format(example.label.lower() in predicted_masks_snippet_rewrite))
            log_file.write("snippet rewrite correct and template incorrect:{}\n".format(
                (example.label.lower() in predicted_masks_snippet_rewrite) and
                (not(example.label.lower() in predicted_template_masks))))
            log_file.write("snippet correct and rewrite incorrect:{}\n\n".format(
                 (not(example.label.lower() in predicted_masks_snippet_rewrite) and
                 (example.label.lower() in predicted_template_masks))))

            log_file.flush()

            if (example.label.lower() in predicted_masks_snippet_rewrite) and not (example.label.lower() in predicted_template_masks):
                snippet_rewrite_better += 1
            elif not (example.label.lower() in predicted_masks_snippet_rewrite) and (example.label.lower() in predicted_template_masks):
                template_better += 1

            if not is_logged:
                not_logged_examples += 1

        if len(args.debug_uuid) > 0:
            continue

        total_more_than_one_mask += more_than_one_mask
        total_no_mask_value += no_mask_value
        total_not_logged_examples += not_logged_examples
        total_skip_sub_label_missing += skip_sub_label_missing
        total_bert_exception += bert_exception
        total_skip_long += skip_long
        total_skip_short += skip_short

        template_precision = template_precision/checked_examples if checked_examples > 0 else 0
        template_precision1 = template_precision1/checked_examples if checked_examples > 0 else 0
        template_mrr = template_mrr/checked_examples if checked_examples > 0 else 0

        snippet_rewrite_precision = snippet_rewrite_precision/checked_examples if checked_examples > 0 else 0
        snippet_rewrite_precision1 = snippet_rewrite_precision1/checked_examples if checked_examples > 0 else 0
        snippet_rewrite_mrr = snippet_rewrite_mrr/checked_examples if checked_examples > 0 else 0

        result_writer.writerow([dataset, relation,
                                len(examples), checked_examples, relation_template,
                                template_precision1, template_precision, template_mrr,
                                snippet_rewrite_precision1, snippet_rewrite_precision, snippet_rewrite_mrr,
                                snippet_rewrite_better,
                                template_better])

        final_realtion_log = "Finished {} {}\n" \
                             "{} examples\n " \
                             "{} checked examples\n " \
                             "{} with a snippet that is invalid for training\n " \
                             "{} with more than one MASK\n " \
                             "{} without MASK token in snippet\n " \
                             "{} without subject in snippet\n " \
                             "{} with snippet too long\n "\
                             "{} with snippet too short\n "\
                             "{} with Bert exception\n " \
                             "{} template_precision1\n" \
                             "{} template_precision\n" \
                             "{} template_mrr\n " \
                             "{} snippet_rewrite_precision1\n " \
                             "{} snippet_rewrite_precision\n" \
                             "{} snippet_rewrite_mrr\n" \
                             "{} snippet_rewrite_better\n" \
                             "{} template_better\n".format(dataset,
                                                           relation,
                                                           len(examples),
                                                           checked_examples,
                                                           not_valid_for_train,
                                                           more_than_one_mask,
                                                           no_mask_value,
                                                           skip_sub_label_missing,
                                                           skip_long,
                                                           skip_short,
                                                           bert_exception,
                                                           not_logged_examples,
                                                           template_precision1, template_precision, template_mrr,
                                                           snippet_rewrite_precision1, snippet_rewrite_precision,
                                                           snippet_rewrite_mrr,
                                                           snippet_rewrite_better,
                                                           template_better,
                                                           )

        print_log(final_realtion_log)
        log_file.write(final_realtion_log)
        log_file.flush()


if __name__ == '__main__':
    args = cli.parse_args()

    logs_pre_path = "BERTese/BERTese/results"
    if args.debug or args.debug_uuid:
        logs_pre_path += "_debug"

    if not os.path.exists(logs_pre_path):
        os.makedirs(logs_pre_path, exist_ok=True)

    data_dir = "BERTese/data"
    if args.debug or args.debug_uuid:
        data_dir += "_debug"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    datasets_file_name = "data.pkl"
    nee_file_name = "nee.pkl"

    if not os.path.exists(os.path.join(data_dir,datasets_file_name)):
        datasets = [("TRex", readers.get_trex_data(args)), ("GoogleRE", readers.get_google_re_data(args))]
        print("Finished reading all examples and writing to file")
        pickle.dump(datasets, open(os.path.join(data_dir, datasets_file_name), "wb"))
    else:
        print("Reading data from pkl file")
        datasets = pickle.load(open(os.path.join(data_dir, datasets_file_name), "rb"))

    if not os.path.exists(os.path.join(data_dir, nee_file_name)):
        nee = NEE()
        for d_type, d_relations in datasets:
            # TODO: CHECK IF WE SWITCH IT PER RELATION IF IT HELPS
            for relation in d_relations:
                print("reading entities for {}, {}".format(d_type, relation))
                for e in d_relations[relation][0]:
                    nee.add_entities(e.snippet)

        print("Finished creating the entities dict")
        pickle.dump(nee, open(os.path.join(data_dir, nee_file_name), "wb"))
    else:
        print("Reading nee from pkl file")
        nee = pickle.load(open(os.path.join(data_dir, nee_file_name), "rb"))

    lama_supervised_experiment(args, datasets, formatter.end_with_mask_random_ner, nee,
                               log_file_path=logs_pre_path + "/nee_random_eos_sentence_log.txt",
                               result_file_path=logs_pre_path + "/nee_random_eos_sentence_results.csv")

    lama_supervised_experiment(args, datasets, formatter.end_with_mask_random_ner, nee,
                               log_file_path=logs_pre_path + "/nee_random_end_mask_sentence_log.txt",
                               result_file_path=logs_pre_path + "/nee_random_end_mask_sentence_results.csv")

    lama_supervised_experiment(args, datasets, formatter.end_with_eos_mask_ner, nee,
                               log_file_path=logs_pre_path + "/nee_eos_sentence_log.txt",
                               result_file_path=logs_pre_path + "/nee_eos_sentence_results.csv")

    lama_supervised_experiment(args, datasets, formatter.end_with_mask_ner, nee,
                               log_file_path=logs_pre_path + "/nee_end_mask_sentence_log.txt",
                               result_file_path=logs_pre_path + "/nee_end_mask_sentence_results.csv")

    lama_supervised_experiment(args, datasets, formatter.eos_mask_sentence, nee,
                               log_file_path=logs_pre_path + "/eos_sentence_log.txt",
                               result_file_path=logs_pre_path + "/eos_sentence_results.csv")

    lama_supervised_experiment(args, datasets, formatter.end_mask_sentence, nee,
                               log_file_path=logs_pre_path + "/end_mask_sentence_log.txt",
                               result_file_path=logs_pre_path + "/end_mask_sentence_results.csv")

    lama_supervised_experiment(args,datasets, formatter.full_masked_sentence, nee,
                               log_file_path=logs_pre_path + "/full_sentence_log.txt",
                               result_file_path=logs_pre_path + "/full_sentence_final_results.csv")