import tqdm
import random
from tqdm import tqdm, trange
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime
import faiss
import random

if not torch.cuda.is_available():
    import BERTese.training_utils as utils
    import BERTese.bert_utils as bert_utils
    from BERTese.utils.logging_utils import print_log_to_screen, print_log_to_file, print_log_to_screen_and_file
    from BERTese.mask_prediction import predict_mask_token as vanilla_bert_mask_prediction
    from BERTese.bert_predictor import init_vanilla_mask_prediction_model
else:
    from utils.logging_utils import print_log_to_screen, print_log_to_file, print_log_to_screen_and_file
    import training_utils as utils
    import bert_utils as bert_utils
    from mask_prediction import predict_mask_token as vanilla_bert_mask_prediction
    from bert_predictor import init_vanilla_mask_prediction_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# Train data and test data is a list of BERTese examples
def train_and_eval_lama(args, pre_trained_model, train_examples, dev_examples, test_examples,
                        training_in_sentence_formatter, inference_in_sentance_formatter,
                        nee, tokenizer, vanilla_bert_model=None):
    print("Training... ")
    is_bertese_model = "bertese" in args.model_type
    acc_method = utils.single_token_simple_accuracy
    model = pre_trained_model
    vocab_size = model.config.vocab_size

    device, n_gpu = utils.get_device()
    # init NN
    nbrs = None
    if is_bertese_model:
        vocab = torch.tensor(list(range(vocab_size)))
        nbrs = model.get_bert_embeddings(vocab)
        nbrs = nbrs.detach().numpy()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    num_train_steps = int(
        len(train_examples) / train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    model.to(device)

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    train_dataloader, train_dataset = load_lama_examples_to_device(args, device, training_in_sentence_formatter,
                                                                   model, tokenizer, train_examples, nee)
    inference_train_dataloader, inference_train_dataset = load_lama_examples_to_device(args, device,
                                                                                       inference_in_sentance_formatter,
                                                                                       model, tokenizer, train_examples, nee)
    dev_dataloader, dev_dataset = load_lama_examples_to_device(args, device, inference_in_sentance_formatter, model,
                                                               tokenizer, dev_examples, nee)
    if args.lpaqa:
        test_dataloader, test_dataset = dev_dataloader, dev_dataset
    else:
        test_dataloader, test_dataset = load_lama_examples_to_device(args, device, inference_in_sentance_formatter,
                                                                     model, tokenizer, test_examples, nee)
    if args.log_tensorx:
        tb_writer = SummaryWriter(args.output_dir)
    else:
        tb_writer = None

    # initial evalutation
    best_dev_acc, best_epoch, best_steps, results_dev = epoch_eval(acc_method, args, 0, 0, dev_dataset, device, 0, 0,
                                                                   model,
                                                                   n_gpu, nbrs, optimizer, scheduler, tokenizer,
                                                                   train_dataset, vanilla_bert_model, 0, vocab_size,
                                                                   tb_writer=tb_writer)
    # Train!

    print_log_to_screen("***** Running training *****")
    print_log_to_screen("  Num examples = {}".format(len(train_dataset)))
    print_log_to_screen("  Num Epochs = {}".format(args.num_train_epochs))
    print_log_to_screen("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))

    print_log_to_screen("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    print_log_to_screen("  Total optimization steps = {}".format(t_total))

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_train_acc = 0.0
    best_steps = 0
    best_epoch = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for i_e, e in enumerate(train_iterator):

        epoch_iterator = tqdm(train_dataloader, desc="Epoch {} Iteration".format(i_e),
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            b_in_tensor = batch[0]
            b_label_tensor = batch[1]

            explicit_mask_loss = 0
            tokens_dist_loss = 0
            if is_bertese_model:
                outputs = model(input_ids=b_in_tensor, mask_label=b_label_tensor)
                if args.tokens_dist_loss_weight > 0:
                    tokens_dist_loss = outputs[1]
                if args.sep_loss_weight > 0:
                    sep_loss = outputs[2]
                if args.explicit_mask_loss_weight > 0:
                    explicit_mask_loss = outputs[3]

            else:
                if args.vocab_gamble_softmax:
                    r = torch.tensor(0.00001)
                    changing_temperature = max(torch.tensor(0.5), 10 * torch.exp(-r * 2 * i_e * 9741))
                    outputs = model(input_ids=b_in_tensor, masked_lm_labels=b_label_tensor,
                                    temperature=changing_temperature)
                else:
                    outputs = model(input_ids=b_in_tensor, masked_lm_labels=b_label_tensor)

            label_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            #loss = (1 - (args.explicit_mask_loss_weight + args.tokens_dist_loss_weight + args.sep_loss_weight)) * loss + \
            #       args.explicit_mask_loss_weight * explicit_mask_loss + \
            #       args.tokens_dist_loss_weight * tokens_dist_loss * args.tokens_dist_scale # + \
            #       args.sep_loss_weight * sep_loss


            loss = (args.label_loss_weight * label_loss)  \
                   + (args.explicit_mask_loss_weight * explicit_mask_loss) \
                   + (args.tokens_dist_loss_weight * tokens_dist_loss)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            #model.zero_grad()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            else:
                continue

            if args.local_rank in [-1, 0] and \
                    args.logging_steps > 0 and global_step % args.logging_steps == 0:

                # Evaluate and Log metrics
                if not args.no_intermediate_evaluation and args.evaluate_during_training:

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank > -1:
                        continue

                    train_results = evaluate_lama(args, vocab_size, model, inference_train_dataset, n_gpu, device,
                                                  tokenizer, e, global_step,
                                                  "Train", acc_method=acc_method,
                                                  sampling_prob=args.train_eval_sampling_prob,
                                                  eval_step="Intermediate Epoch {}".format(i_e), nbrs=nbrs,
                                                  vanilla_bert=vanilla_bert_model,
                                                  examples_log_prob=args.intermediate_examples_log_prob)
                    if args.log_tensorx:
                        for key, value in train_results.items():
                            tb_writer.add_scalar("train_{}".format(key), value, global_step)

                    if "eval_acc" in train_results and train_results["eval_acc"] > best_train_acc:
                        log_msg = "Improved (Intermediate) Train acc: {}, loss: {}," \
                                  "label loss:{}, mask loss:{}, sep loss:{}, tokens_dist_loss:{}, global steps: {}". \
                            format(train_results["eval_acc"],
                                   train_results["eval_loss"],
                                   train_results["eval_label_loss"],
                                   train_results["eval_mask_loss"],
                                   train_results["eval_sep_loss"],
                                   train_results["eval_tokens_dist_loss"],
                                   global_step)
                        print_log_to_screen(log_msg)
                        best_train_acc = train_results["eval_acc"]

                        # Evaluate Dev (we do that only if we have a better accuracy on the training set
                        if args.do_eval_dev:
                            dev_results = evaluate_lama(args, vocab_size, model, dev_dataset, n_gpu, device, tokenizer,
                                                        e, global_step,
                                                        "Dev", acc_method=acc_method,
                                                        sampling_prob=args.test_eval_sampling_prob,
                                                        eval_step="Intermediate Epoch {}".format(i_e), nbrs=nbrs,
                                                        vanilla_bert=vanilla_bert_model,
                                                        examples_log_prob=args.intermediate_examples_log_prob)
                            if args.log_tensorx:
                                for key, value in dev_results.items():
                                    tb_writer.add_scalar("dev_{}".format(key), value, global_step)

                            dev_log_msg = "Intermediate Dev acc: {},loss: {}, " \
                                          "label loss:{}, mask loss:{}, sep loss:{}, tokens dist loss:{}, global steps: {}". \
                                format(results_dev["eval_acc"],
                                       results_dev["eval_loss"],
                                       results_dev["eval_label_loss"],
                                       results_dev["eval_mask_loss"],
                                       results_dev["eval_sep_loss"],
                                       results_dev["eval_tokens_dist_loss"],
                                       global_step)

                            if "eval_acc" in dev_results and dev_results["eval_acc"] > best_dev_acc:
                                dev_log_msg = "Improved " + dev_log_msg
                                best_dev_acc = dev_results["eval_acc"]
                                best_epoch = e
                                best_steps = global_step
                                utils.save_model(args, global_step, model, optimizer, scheduler, tokenizer,
                                                 print_log_to_screen)

                            print_log_to_screen(dev_log_msg)

                if args.log_tensorx:
                    tb_writer.add_scalar("Train lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("Train loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar("Train vocab loss", (tokens_dist_loss - logging_loss) / args.logging_steps, global_step)

                if args.screen_logs:
                    print_log_to_screen(
                        "Average loss: {} at global step: {}".format(str((tr_loss - logging_loss) / args.logging_steps),
                                                                     str(global_step)))
                logging_loss = tr_loss

            if (args.max_steps > 0) and (global_step > args.max_steps):
                epoch_iterator.close()
                break

        # eval after EPOCH
        if args.evaluate_during_training:
            best_dev_acc, best_epoch, best_steps, results_dev = epoch_eval(acc_method, args, best_dev_acc, best_steps,
                                                                           dev_dataset, device, e, global_step, model,
                                                                           n_gpu, nbrs, optimizer, scheduler, tokenizer,
                                                                           inference_train_dataset, vanilla_bert_model,
                                                                           best_epoch, vocab_size,tb_writer)


        else:  # we either save the model after the acc improved or after every epoch
            utils.save_model(args, global_step, model, optimizer, scheduler, tokenizer, print_log_to_screen)

        if (args.max_steps > 0) and (global_step > args.max_steps):
            train_iterator.close()
            break

    # Final Evaluation
    dev_out, test_out, train_out = final_evaluation(acc_method, args, best_epoch, best_steps, dev_dataset, device, e,
                                                    global_step, inference_train_dataset, model, n_gpu, nbrs, tb_writer,
                                                    test_dataset, tokenizer, vanilla_bert_model, vocab_size)

    return model, train_out, dev_out, test_out


def final_evaluation(acc_method, args, best_epoch, best_steps, dev_dataset, device, e, global_step,
                     inference_train_dataset, model, n_gpu, nbrs, tb_writer, test_dataset, tokenizer,
                     vanilla_bert_model, vocab_size):
    final_results_train = evaluate_lama(args, vocab_size, model, inference_train_dataset, n_gpu, device, tokenizer, e,
                                        global_step, "Train",
                                        acc_method=acc_method, sampling_prob=1, eval_step="Final", nbrs=nbrs,
                                        vanilla_bert=vanilla_bert_model, examples_log_prob=1)
    if "eval_acc" in final_results_train:
        print_log_to_screen("Final results: Epoch: {}, Train acc: {}, loss: {}, label loss:{}, mask loss:{}, "
                            "sep loss:{}, tokens dist loss: {}\n".
                            format(e, final_results_train["eval_acc"],
                                   final_results_train["eval_loss"],
                                   final_results_train["eval_label_loss"],
                                   final_results_train["eval_mask_loss"],
                                   final_results_train["eval_sep_loss"],
                                   final_results_train["eval_tokens_dist_loss"]))
    if args.do_eval_dev:
        final_results_dev = evaluate_lama(args, vocab_size, model, dev_dataset, n_gpu, device, tokenizer, e,
                                          global_step, "Dev",
                                          acc_method=acc_method, sampling_prob=1, eval_step="Final", nbrs=nbrs,
                                          vanilla_bert=vanilla_bert_model, examples_log_prob=1)
        if "eval_acc" in final_results_dev:
            print_log_to_screen("Final results: Epoch: {}, Dev acc: {}, loss: {}, label loss:{}, mask loss:{},"
                                "sep loss:{}, token dist loss:{}\n".
                                format(e, final_results_dev["eval_acc"],
                                       final_results_dev["eval_loss"],
                                       final_results_dev["eval_label_loss"],
                                       final_results_dev["eval_mask_loss"],
                                       final_results_dev["eval_sep_loss"],
                                       final_results_dev["eval_tokens_dist_loss"]))
    if args.do_eval_test:
        final_results_test = evaluate_lama(args, vocab_size, model, test_dataset, n_gpu, device, tokenizer, e,
                                           global_step, "Test",
                                           acc_method=acc_method, sampling_prob=1, eval_step="Final", nbrs=nbrs,
                                           vanilla_bert=vanilla_bert_model, examples_log_prob=1)
        if "eval_acc" in final_results_test:
            print_log_to_screen("Final results: Epoch: {}, Dev acc: {}, loss: {}, label loss:{}, mask loss:{},"
                                "sep loss:{}, token dist loss:{}\n".
                                format(e, final_results_test["eval_acc"],
                                       final_results_test["eval_loss"],
                                       final_results_test["eval_label_loss"],
                                       final_results_test["eval_mask_loss"],
                                       final_results_test["eval_sep_loss"],
                                       final_results_test["eval_tokens_dist_loss"]))
    if args.local_rank in [-1, 0] and args.log_tensorx:
        tb_writer.close()
    train_out = {"best_steps": best_steps, "best_epoch": best_epoch,
                 "global_steps:": global_step,
                 "final_loss": final_results_train["eval_loss"],
                 "acc": final_results_train["eval_acc"]}
    if "top10_eval_acc" in final_results_train:
        train_out["top10_acc"] = final_results_train["top10_eval_acc"]
    if args.do_eval_dev:
        dev_out = {"final_loss": final_results_dev["eval_loss"],
                   "acc": final_results_dev["eval_acc"]}

        if "top10_eval_acc" in final_results_dev:
            dev_out["top10_acc"] = final_results_dev["top10_eval_acc"]
    else:
        dev_out = {}
    if args.do_eval_test:
        test_out = {"final_loss": final_results_test["eval_loss"],
                    "acc": final_results_test["eval_acc"], "top10_acc": final_results_test["top10_eval_acc"]}
        if "top10_eval_acc" in final_results_test:
            test_out["top10_acc"] = final_results_test["top10_eval_acc"]
    else:
        test_out = {}
    return dev_out, test_out, train_out


def epoch_eval(acc_method, args, best_dev_acc, best_steps, dev_dataset, device, e, global_step, model,
               n_gpu, nbrs, optimizer, scheduler, tokenizer, train_dataset, vanilla_bert_model, best_epoch, vocab_size,
               tb_writer):
    results_train = evaluate_lama(args, vocab_size, model, train_dataset, n_gpu, device, tokenizer, e, global_step,
                                  "Train",
                                  acc_method=acc_method, sampling_prob=1,
                                  eval_step="Epoch", nbrs=nbrs, vanilla_bert=vanilla_bert_model,
                                  examples_log_prob=args.train_examples_log_prob)
    if "eval_acc" in results_train:
        print_log_to_screen("Epoch: {}, Train acc: {}, loss:{}, label loss:{}, mask loss:{}, sep loss:{}"
                            " tokens dist loss:{}\n".
                            format(e, results_train["eval_acc"], results_train["eval_loss"],
                                   results_train["eval_label_loss"],
                                   results_train["eval_mask_loss"],
                                   results_train["eval_sep_loss"],
                                   results_train["eval_tokens_dist_loss"]))

        if args.log_tensorx:
            tb_writer.add_scalar("Train Epoch lr", scheduler.get_lr()[0], e)
            tb_writer.add_scalar("Train Epoch loss", results_train["eval_loss"], e)
            tb_writer.add_scalar("Train vocab loss", results_train["eval_tokens_dist_loss"], e)

    if args.do_eval_dev:
        results_dev = evaluate_lama(args, vocab_size, model, dev_dataset, n_gpu, device, tokenizer, e, global_step,
                                    "Dev",
                                    acc_method=acc_method, sampling_prob=1,
                                    eval_step="Epoch", nbrs=nbrs, vanilla_bert=vanilla_bert_model,
                                    examples_log_prob=args.dev_examples_log_prob)

        if "eval_acc" in results_dev and results_dev["eval_acc"] > best_dev_acc:
            best_dev_acc = results_dev["eval_acc"]
            best_epoch = e
            best_steps = global_step
            utils.save_model(args, global_step, model, optimizer, scheduler, tokenizer,
                             print_log_to_screen)

        if "eval_acc" in results_dev:
            print_log_to_screen("Epoch: {}, Dev acc: {}, loss: {}\n".
                                format(e, results_dev["eval_acc"], results_dev["eval_loss"]))
            if args.log_tensorx:
                tb_writer.add_scalar("Dev Epoch lr", scheduler.get_lr()[0], e)
                tb_writer.add_scalar("Dev Epoch loss", results_dev["eval_loss"], e)
                tb_writer.add_scalar("Dev vocab loss", results_dev["eval_tokens_dist_loss"], e)
    else:
        results_dev = None
    return best_dev_acc, best_epoch, best_steps, results_dev


def load_lama_examples_to_device(args, device, in_sentence_formatter, model, tokenizer, examples, nee):
    is_bertese_model = "bertese" in args.model_type
    print_log_to_screen("Loading to device {} examples".format(len(examples)))
    # Encode the input to the encoder (the question)
    in_seqs = []
    masked_lm_labels_ids = []
    label_ids = []

    valid_ex_ct = 0
    valid_identity_items = 0
    for example_index, example in enumerate(examples):
        if args.debug and \
           args.debug_single_item_index >= 0 and \
           args.debug_single_item_index != example_index:
            continue

        label = example.label
        label_id = tokenizer.encode(label, add_special_tokens=False)
        if len(label_id) > 1:
            continue
        label_ids.append(label_id)
        masked_sentence, _ = in_sentence_formatter(example, nee)

        if args.random_input:
            input_ids = [random.randrange(1037, tokenizer.vocab_size, 1) for i in range(args.max_seq_length)]
            if not args.dont_include_special_tokens:
                input_ids = [101] + input_ids + [102]
        else:
            input_ids = tokenizer.encode(masked_sentence,
                                         pad_to_max_length=True,
                                         max_length=args.max_seq_length,
                                         add_special_tokens=not args.dont_include_special_tokens)
        in_seqs.append(input_ids)

        if "bertese" not in args.model_type:
            unmasked_sentence = example.unmasked_sentence.lower() if "cased" in args.upper_model_name \
                else examples.unmasked_sentence

            masked_lm_labels_id = tokenizer.encode(unmasked_sentence,
                                                   pad_to_max_length=True,
                                                   max_length=args.max_seq_length, )

            masked_lm_labels_ids.append(masked_lm_labels_id)

        valid_ex_ct += 1
    if args.debug_identity_examples:
        print_log_to_screen("You have {} identity items to debug".format(len(in_seqs)))

    assert (len(in_seqs) > 0)
    # Convert inputs to PyTorch tensors
    in_seq_tensor = in_seqs if "ignore" in args.model_type else torch.tensor(in_seqs)
    labels_tensor = label_ids if "ignore" in args.model_type else torch.tensor(label_ids)

    in_seq_tensor = in_seq_tensor.to(device)
    labels_tensor = labels_tensor.to(device)

    if is_bertese_model:
        dataset = TensorDataset(in_seq_tensor, labels_tensor)
    else:
        masked_lm_labels_tensor = masked_lm_labels_ids if "ignore" in args.model_type else torch.tensor(
            masked_lm_labels_ids)
        masked_lm_labels_tensor = masked_lm_labels_tensor.to(device)

        dataset = TensorDataset(in_seq_tensor, masked_lm_labels_tensor)
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size)
    print_log_to_screen("Loaded {0} examples out of {1}".format(valid_ex_ct, len(examples)))

    return dataloader, dataset


def evaluate_lama(args, vocab_size, model, eval_dataset, n_gpu, device, tokenizer, epoch, global_steps, eval_set_name,
                  acc_method=utils.simple_accuracy, topk_acc_method=utils.topk_simple_accuracy, sampling_prob=1,
                  eval_step="", nbrs=None, upper_model_bert="bert-base-uncased", vanilla_bert=None,
                  examples_log_prob=0.1):
    """
        eval_set_name: Train, Dev, Test
        eval_step: intermediate, epoch, final
    """

    is_bertese_model = "bertese" in args.model_type
    screen_logs = args.screen_logs and random.random() < args.logging_sampling_prob
    results = {}
    eval_output_dir = args.output_dir

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    output_eval_file = os.path.join(eval_output_dir, args.model_type + "_" + eval_set_name + "_eval_log.txt")
    summery_log_file = os.path.join(eval_output_dir, args.model_type + "_" + eval_set_name + "_eval_summery.txt")

    log_writter = open(output_eval_file, "a+")
    summery_writter = open(summery_log_file, "a+")

    # Eval!
    """
    if screen_logs :
        print_log_to_screen("***** Running {} evaluation on {} *****".format(eval_step, eval_set_name))
        print_log_to_screen("  Num examples = {} out of a total of {}".format(round(len(eval_dataset)*sampling_prob), len(eval_dataset)))
        print_log_to_screen("  Batch size = {}".format(args.eval_batch_size))
    """

    eval_loss = 0.0
    eval_label_loss = 0.0
    eval_mask_loss = 0.0
    eval_sep_loss = 0.0
    eval_tokens_dist_loss = 0.0
    nb_eval_steps = 0
    eval_accuracy = 0
    top10_eval_accuracy = 0
    print_log_to_file(log_writter,
                      "\n----- " + eval_set_name + " epoch: {}, global steps: {}-----\n".format(epoch, global_steps))
    print_log_to_file(summery_writter,
                      "\n----- " + eval_set_name + " epoch: {}, global steps: {}-----\n".format(epoch, global_steps))

    if screen_logs:
        iter_set = tqdm(eval_dataloader, desc="Evaluating " + eval_set_name)
    else:
        iter_set = eval_dataloader
    for batch in iter_set:
        model.eval()
        if random.random() > sampling_prob:
            continue
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            b_in_tensor = batch[0]
            b_label_tensor = batch[1]

            if is_bertese_model:
                tmp_label_eval_loss, tokens_dist_loss, sep_loss, explicit_mask_loss, mask_dist_softmin, mask_dist, label_logits, lower_models_last_hs, upper_model_logits = \
                    model(b_in_tensor, b_label_tensor)

                # tmp_eval_loss = (1 - (args.explicit_mask_loss_weight + args.tokens_dist_loss_weight + args.sep_loss_weight)) * tmp_label_eval_loss + \
                tmp_eval_loss = (args.label_loss_weight * tmp_label_eval_loss) + \
                                (args.explicit_mask_loss_weight * explicit_mask_loss) + \
                                (args.tokens_dist_loss_weight * tokens_dist_loss) # * args.tokens_dist_scale \
                                #+ \
                                #args.sep_loss_weight * sep_loss

                batch_label_loss = tmp_label_eval_loss.mean().item()
                batch_mask_loss = explicit_mask_loss.mean().item()
                batch_tokens_dist_loss = tokens_dist_loss.mean().item()
                batch_sep_loss = sep_loss.mean().item()

            else:  # this is bert
                tmp_eval_loss, label_logits = \
                    model(b_in_tensor, masked_lm_labels=b_label_tensor)
            batch_eval_loss = tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        preds_logits = label_logits.detach().cpu().numpy()
        label_ids = b_label_tensor.detach().cpu().numpy()
        input_seq_ids = b_in_tensor.detach().cpu().numpy()
        sort_axis = 0 if len(preds_logits) == vocab_size else 1

        if not is_bertese_model:
            mask_idxs = np.argmax(input_seq_ids == bert_utils.MASK_ID, axis=sort_axis)
            preds_logits = np.array([preds_logits[i, mask_idxs[i], :] for i in range(len(mask_idxs))])
            label_ids = np.array([label_ids[i, mask_idxs[i]] for i in range(len(mask_idxs))])

        if is_bertese_model:
            max_softmin_mask_dist = mask_dist_softmin.max(dim=sort_axis)[0]
            min_mask_dist = mask_dist.min(dim=sort_axis)[0]

        pred_arr = np.argmax(preds_logits, axis=sort_axis)

        if pred_arr.size == 1:
            pred_top10_arr = preds_logits.argsort(axis=sort_axis)[-10:]
            pred_arr = [pred_arr]
            pred_top10_arr = [pred_top10_arr]
        else:
            pred_top10_arr = preds_logits.argsort(axis=sort_axis)[:, -10:]

        if nbrs is not None:
            rewrite_seq_embes = lower_models_last_hs.detach().cpu().numpy()
            rewrite_seq_ids = utils.get_nearset_token(preds=rewrite_seq_embes, nbrs=nbrs)

        preds = []
        top10_preds = []
        labels = []
        input_tokens = []
        for i in range(len(pred_arr)):
            preds.append(tokenizer.decode([pred_arr[i]], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            input_tokens.append(tokenizer.decode(input_seq_ids[i][input_seq_ids[i] != 0], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=True))
            if is_bertese_model:
                top10_preds.append(
                    tokenizer.decode(pred_top10_arr[i], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            labels.append(
                tokenizer.decode([label_ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=True))

            if args.screen_logs and args.log_examples:
                if random.random() > examples_log_prob:
                    continue

                x = bert_utils.bert_decode_clean(input_seq_ids[i], tokenizer)
                print_log_to_screen_and_file(log_writter, "\nX: {}".format(x))

                if is_bertese_model:
                    clean_x_rewrite = bert_utils.bert_decode_clean(rewrite_seq_ids[i], tokenizer)
                    #if args.debug:
                    #    tokens = []
                    #    for s in rewrite_seq_ids[i]:
                    #        tokens.append(tokenizer.decode([s], skip_special_tokens=False, clean_up_tokenization_spaces=True))
                    #    print_log_to_screen(tokens)
                    print_log_to_screen_and_file(log_writter, "Clean X LB rewrite (NN): {}".format(clean_x_rewrite))
                    no_ped_x_rewrite = tokenizer.decode(rewrite_seq_ids[i][rewrite_seq_ids[i] != 0])
                    print_log_to_screen_and_file(log_writter, "X LB rewrite (NN): {}".format(no_ped_x_rewrite))

                    print_log_to_screen_and_file(log_writter, "Same rewrite? {}".format(
                        x.replace(".", "") == clean_x_rewrite.replace(".", "")))

                y = tokenizer.decode([label_ids[i]])
                print_log_to_screen_and_file(log_writter, "Y: {}".format(y))

                y_hat = tokenizer.decode([pred_arr[i]])
                print_log_to_screen_and_file(log_writter, "PRED: {}".format(y_hat))

                print_log_to_screen_and_file(log_writter, "Y == PRED?: {}".format(y == y_hat))

                if is_bertese_model:
                    try:
                        if args.optimize_mask_softmin:
                            print_log_to_screen_and_file(log_writter, "Max MASK Distance Softmin Score:{}".
                                                         format(max_softmin_mask_dist[i]))
                        else:
                            print_log_to_screen_and_file(log_writter, "Min MASK Distance:{}".format(min_mask_dist[i]))
                    except:
                        print("error:")
                        print("min_mask_dist len ", len(min_mask_dist))
                        print("pred_arr len", len(pred_arr))
                    try:
                        y_bert = \
                            vanilla_bert_mask_prediction(model.upper_bert, tokenizer, x, pred_count=0,
                                                         add_cls_sep=False)[0]
                        if vanilla_bert is not None:
                            y_bert_2 = \
                                vanilla_bert_mask_prediction(vanilla_bert, tokenizer, x, pred_count=0,
                                                             add_cls_sep=False)[0]
                            # print_log_to_screen("BERT_PRED VANILLA:{}".format(y_bert_2))
                            assert (y_bert == y_bert_2)
                    except ValueError:
                        print_log_to_screen_and_file(log_writter, "MORE THAN ONE MASK VALUE IN INPUT: {}".format(x))
                        y_bert = ""
                    print_log_to_screen_and_file(log_writter, "BERT_PRED: {}".format(y_bert))
                    print_log_to_screen_and_file(log_writter, "Y_BERT == PERD?: {}".format(y_bert == y_hat))
                    print_log_to_screen_and_file(log_writter, "Nailed it?: {}".format(y == y_hat and y_hat != y_bert))
                    print_log_to_screen_and_file(log_writter, "TOP10 PRED: {}".format(
                        [tokenizer.decode([token]) for token in pred_top10_arr[i]]))

        batch_acc = acc_method(preds, labels, do_logs=False, log_writer=log_writter)
        if is_bertese_model:
            batch_top10_acc = topk_acc_method(top10_preds, labels, do_logs=False, log_writer=log_writter)
            top10_eval_accuracy += batch_top10_acc
        eval_accuracy += batch_acc
        eval_loss += batch_eval_loss
        if is_bertese_model:
            eval_label_loss += batch_label_loss
            eval_mask_loss += batch_mask_loss
            eval_tokens_dist_loss += batch_tokens_dist_loss

        log_msg = "\n----- " + eval_set_name + " epoch: {}, steps: {}, batch loss: {}, batch acc: {}". \
            format(epoch, nb_eval_steps, batch_eval_loss, batch_acc)
        if is_bertese_model:
            log_msg += ", batch top 10 batch acc:{}".format(batch_top10_acc)
            log_msg += ", batch label loss: {}".format(batch_label_loss)
            log_msg += ", batch mask loss: {}".format(batch_mask_loss)
        log_msg += "-----\n"

        print_log_to_file(log_writter, log_msg)

        result = {"batch_eval_acc": batch_acc, "batch_eval_loss": eval_loss}
        if is_bertese_model:
            result["batch_label_loss"] = batch_label_loss
            result["batch_mask_loss"] = batch_mask_loss
            result["batch_top10_acc"] = batch_top10_acc
            result["batch_sep_loss"] = batch_sep_loss
            result["batch_tokens_dist_loss"] = batch_tokens_dist_loss
        else:
            result["batch_label_loss"] = "N/A"
            result["batch_mask_loss"] = "N/A"
            result["batch_top10_acc"] = "N/A"
            result["batch_sep_loss"] = "N/A"
            result["batch_tokens_dist_loss"] = "N/A"
        results.update(result)

    if nb_eval_steps > 0:
        eval_accuracy /= nb_eval_steps
        top10_eval_accuracy /= nb_eval_steps
        eval_loss /= nb_eval_steps
        if is_bertese_model:
            eval_mask_loss /= nb_eval_steps
            eval_label_loss /= nb_eval_steps
            eval_sep_loss /= nb_eval_steps
            eval_tokens_dist_loss /= nb_eval_steps

    log_msg = "\n----- " + eval_set_name + " epoch: {}, steps: {},  loss: {}, acc: {}". \
        format(epoch, nb_eval_steps, eval_loss, eval_accuracy)
    if is_bertese_model:
        log_msg += ", top10 acc: {}".format(top10_eval_accuracy)
        log_msg += ", label loss: {}".format(eval_label_loss)
        log_msg += ", mask loss: {}".format(eval_mask_loss)
        log_msg += ", sep loss: {}".format(eval_sep_loss)
        log_msg += ", tokens dist loss: {}".format(eval_tokens_dist_loss)
    else:
        log_msg += ", top10 acc: N\A"
        log_msg += ", label loss: N\A"
        log_msg += ", mask loss: N\A"
        log_msg += ", sep loss: N\A"
        log_msg += ", tokens dist loss: N\A"
    log_msg += " -----\n"

    print_log_to_screen_and_file(log_writter, log_msg)
    print_log_to_file(summery_writter, log_msg)

    summery_writter.flush()
    log_writter.flush()

    summery_writter.close()
    log_writter.close()

    out = {"eval_acc": eval_accuracy, "eval_loss": eval_loss}
    if is_bertese_model:
        out["top10_eval_acc"] = top10_eval_accuracy
        out["eval_label_loss"] = eval_label_loss
        out["eval_mask_loss"] = eval_mask_loss
        out["eval_sep_loss"] = eval_sep_loss
        out["eval_tokens_dist_loss"] = eval_tokens_dist_loss
    else:
        out["top10_eval_acc"] = "N/A"
        out["eval_label_loss"] = "N/A"
        out["eval_mask_loss"] = "N/A"
        out["eval_sep_loss"] = "N/A"
        out["eval_tokens_dist_loss"] = "N/A"
    return out
