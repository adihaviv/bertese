import tqdm
import random
from tqdm import tqdm, trange
import numpy as np
import os
import torch
from transformers import BertForMaskedLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

if not torch.cuda.is_available():
    import BERTese.training_utils as utils
    from BERTese.utils.logging_utils import print_log_to_screen, print_log_to_file
    import BERTese.bert_utils as bert
    import BERTese.models.t5_models as t5
else:
    from utils.logging_utils import print_log_to_screen, print_log_to_file
    import training_utils as utils
    import bert_utils as bert
    import models.t5_models as t5


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# Train data and test data is a list of BERTese examples
def train_and_eval(args, pre_trained_seq2seq_model, train_examples, dev_examples, test_examples,
                   in_sentence_formatter, out_sentence_formatter, nee, tokenizer, vocab_size):
    print("Training... ")
    out_dir = args.model_type + "_" + str(datetime.now().strftime("%d_%m_%H_%M_%S"))
    if args.debug:
        out_dir = "debug_" + out_dir
    args.output_dir = os.path.join(args.output_dir, out_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.acc_method.lower() == "exact":
        acc_method = utils.simple_accuracy
    elif args.acc_method.lower() == "bleu":
        acc_method = utils.bleu_accuracy
    elif args.acc_method.lower() == "after_upper_model_bleu_accuracy":
        acc_method = utils.after_upper_model_bleu_accuracy
    elif args.acc_method.lower() == "t5_exact":
        acc_method = utils.t5_exact
    else:
        print_log_to_screen("INVALID ACC METHOD. USING EXACT")
        acc_method = utils.simple_accuracy

    model = pre_trained_seq2seq_model
    device, n_gpu = utils.get_device()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    num_train_steps = int(len(train_examples) / train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
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

    train_dataloader, train_dataset = load_examples_to_device(args, device, in_sentence_formatter, out_sentence_formatter, tokenizer, train_examples, nee, model)
    dev_dataloader, dev_dataset = load_examples_to_device(args, device, in_sentence_formatter, out_sentence_formatter,
                                                            tokenizer, dev_examples, nee, model)
    test_dataloader, test_dataset = load_examples_to_device(args, device, in_sentence_formatter, out_sentence_formatter, tokenizer, test_examples, nee, model)

    # Train!
    tb_writer = SummaryWriter()

    print_log_to_screen("***** Running training *****")
    print_log_to_screen("  Num examples = {}".format(len(train_dataset)))
    print_log_to_screen("  Num Epochs = {}".format(args.num_train_epochs))
    print_log_to_screen(
        "  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))

    print_log_to_screen("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    print_log_to_screen("  Total optimization steps = {}".format(t_total))

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    for e in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            b_in_tensor = batch[0]
            b_out_tensor = batch[1]
            b_label_tensor = batch[2]
            if "bert_emb" in args.model_type:
                outputs = model(input_ids=b_in_tensor, lm_labels=b_label_tensor)
            elif "t5_emb" in args.model_type:
                outputs = model(input_ids=b_in_tensor, decoder_lm_labels=b_label_tensor)
            elif "t5" in args.model_type:
                outputs = model(input_ids=b_in_tensor, decoder_lm_labels=b_label_tensor)
            else:
                outputs = model(b_in_tensor, b_out_tensor, decoder_lm_labels=b_label_tensor)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            else:
                continue

            # Log metrics
            if args.no_intermediate_evaluation:
                continue
            elif args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step % args.logging_steps == 0):
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    train_results = evaluate(args, model, train_dataset, n_gpu, device, tokenizer, e, global_step,
                                             "Train", acc_method=acc_method, sampling_prob=args.train_eval_sampling_prob,
                                             vocab_size=vocab_size)
                    for key, value in train_results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    if "eval_acc" in train_results and train_results["eval_acc"] > best_dev_acc:
                        if args.intermediate_screen_logs:
                            log_msg = "Intermediate Train acc: {}, loss: {}, global steps: {}".format(train_results["eval_acc"],
                                                                               train_results["eval_loss"],
                                                                               global_step)
                            print_log_to_screen(log_msg)
                        best_dev_acc = train_results["eval_acc"]
                        best_steps = global_step

                        # Evaluate Dev
                        if args.evaluate_during_training and args.do_eval_dev:
                            results_test = evaluate(args, model, dev_dataset, n_gpu, device, tokenizer, e, global_step,
                                                    "Dev", acc_method=acc_method, sampling_prob=args.test_eval_sampling_prob,
                                                    vocab_size=vocab_size)
                            if args.intermediate_screen_logs:
                                for key, value in results_test.items():
                                    tb_writer.add_scalar("test_{}".format(key), value, global_step)
                                print_log_to_screen("Intermediate Dev acc: {}, loss: {}, global steps: {}".format(results_test["eval_acc"],results_test["eval_loss"],global_step))

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                if args.intermediate_screen_logs:
                    print_log_to_screen("Average loss: {} at global step: {}".format(str((tr_loss - logging_loss) / args.logging_steps),str(global_step)))
                logging_loss = tr_loss

                utils.save_model(args, global_step, model, optimizer, scheduler, tokenizer, print_log_to_screen)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        #eval after EPOCH
        if args.evaluate_during_training:
            results_train = evaluate(args, model, train_dataset, n_gpu, device, tokenizer, e,
                                     global_step, "Train", acc_method=acc_method,
                                     sampling_prob=args.train_eval_sampling_prob, vocab_size=vocab_size)
            if "eval_acc" in results_train:
                print_log_to_screen("Epoch: {}, Train acc: {}, loss: {}\n".format(e, results_train["eval_acc"], results_train["eval_loss"]))
            if args.do_eval_dev:
                results_dev = evaluate(args, model, dev_dataset, n_gpu, device, tokenizer,e,
                                       global_step, "Dev", acc_method=acc_method,
                                       sampling_prob=args.train_eval_sampling_prob, vocab_size=vocab_size)
                if "eval_acc" in results_dev:
                    print_log_to_screen("Epoch: {}, Dev acc: {}, loss: {}\n".format(e, results_dev["eval_acc"], results_dev["eval_loss"]))

        # save after epoch
        utils.save_model(args, global_step, model, optimizer, scheduler, tokenizer, print_log_to_screen)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # Final Evaluation
    results_train = evaluate(args, model, train_dataset, n_gpu, device, tokenizer, e, global_step, "Train",
                             acc_method=acc_method, sampling_prob=1,vocab_size=vocab_size)
    if "eval_acc" in results_train:
        print_log_to_screen("Final results: Epoch: {}, Train acc: {}, loss: {}\n".format(e, results_train["eval_acc"], results_train["eval_loss"]))

    if args.do_eval_dev:
        results_dev = evaluate(args, model, dev_dataset, n_gpu, device, tokenizer, e, global_step, "Dev",
                               acc_method=acc_method, sampling_prob=1,vocab_size=vocab_size)
        if "eval_acc" in results_test:
            print_log_to_screen("Final results: Epoch: {}, Test acc: {}, loss: {}\n".format(e, results_dev["eval_acc"], results_dev["eval_loss"]))

    if args.do_eval_test:
        results_test = evaluate(args, model, test_dataset, n_gpu, device, tokenizer, e, global_step, "Test",
                                acc_method=acc_method, sampling_prob=1,vocab_size=vocab_size)
        if "eval_acc" in results_test:
            print_log_to_screen("Final results: Epoch: {}, Test acc: {}, loss: {}\n".format(e, results_test["eval_acc"], results_test["eval_loss"]))

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model, global_step, tr_loss / global_step, best_steps


def load_examples_to_device(args, device, in_sentence_formatter, out_sentence_formatter,
                            tokenizer, train_examples, nee, model=None):
    # Encode the input to the encoder (the question)
    trained_examples_ct = len(train_examples)
    print_log_to_screen("Loading to device {} examples".format(trained_examples_ct))
    in_seqs = []
    out_seqs = []

    for example in train_examples:
        out_masked_sentence, valid_for_train = out_sentence_formatter(example, nee)
        example.valid_for_train = valid_for_train

        if not valid_for_train:
            trained_examples_ct -= 1
            continue

        ## Run validation tests
        if len(out_masked_sentence.split()) > 100 or \
           len(out_masked_sentence.split()) < 1 or \
            out_masked_sentence.count(bert.MASK) > 1 or \
            out_masked_sentence.count(bert.MASK) < 1:

            trained_examples_ct -= 1
            continue


        if len(out_masked_sentence.split()) > 80 or \
                len(out_masked_sentence.split()) < 1 or \
                out_masked_sentence.count(bert.MASK) > 1 or \
                out_masked_sentence.count(bert.MASK) < 1:
            trained_examples_ct -= 1
            continue

        if "t5_emb" in args.model_type:
            lm_labels = model.bert_tokenizer.encode(out_masked_sentence,
                                                    pad_to_max_length=True,
                                                    max_length=args.max_seq_length,)
        else:
            #lm_labels = lower_bert.bert_encode(out_masked_sentence, tokenizer)
            #padding = [0] * (args.max_seq_length - len(lm_labels))
            #lm_labels += padding

            #todo: we might need to change to the comment becouse of the double [CLS]
            lm_labels = tokenizer.encode(out_masked_sentence,
                                         pad_to_max_length=True,
                                         max_length=args.max_seq_length,)
        out_seqs.append(lm_labels)

        ## Run validation tests
        #todo: refactor, need to pass in the sentence and not just the example. only problematic for the snippets!
        template_masked_sentence, _ = in_sentence_formatter(example,nee)
        if template_masked_sentence.count(bert.MASK) > 1 or \
           template_masked_sentence.count(bert.MASK) < 1:
            trained_examples_ct -= 1
            continue

        #in_seq_ids = lower_bert.bert_encode(template_masked_sentence, tokenizer)
        #padding = [0] * (args.max_seq_length - len(in_seq_ids))
        #in_seq_ids += padding
        #in_seqs.append(in_seq_ids)

        if "t5" in args.model_type:
            template_masked_sentence.replace(bert.MASK, t5.T5_MASK)
        input_ids = tokenizer.encode(template_masked_sentence,
                                     pad_to_max_length=True,
                                     max_length=args.max_seq_length,)
        in_seqs.append(input_ids)

    # Convert inputs to PyTorch tensors
    in_seq_tensor = in_seqs if "ignore" in args.model_type else torch.tensor(in_seqs)
    out_seq_tensor = out_seqs if "ignore" in args.model_type else torch.tensor(out_seqs)
    lm_labels = out_seqs
    labels_tensor = lm_labels if "ignore" in args.model_type else torch.tensor(out_seqs)
    in_seq_tensor = in_seq_tensor.to(device)
    out_seq_tensor = out_seq_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    train_dataset = TensorDataset(in_seq_tensor, out_seq_tensor, labels_tensor)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader, train_dataset


def evaluate(args, model, eval_dataset, n_gpu, device, tokenizer, epoch, global_steps, eval_type,
             acc_method=utils.bleu_accuracy, sampling_prob=1, vocab_size=0):

    results = {}
    eval_output_dir = args.output_dir

    if "emb" in args.model_type:
        with torch.no_grad():
            vocab = torch.tensor(list(range(vocab_size))).to(device)
            nbrs = model.bert.embeddings.word_embeddings(vocab).detach().cpu().numpy()
    if acc_method == utils.after_upper_model_bleu_accuracy:
        upper_model = BertForMaskedLM.from_pretrained('bert-large-uncased' if 'bert-large' in args.model_name else 'bert-base-uncased')
            # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(embs.cpu())
    #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    output_eval_file = os.path.join(eval_output_dir, args.model_type+"_"+eval_type+"_eval_log.txt")
    summery_log_file = os.path.join(eval_output_dir, args.model_type+"_"+eval_type+"_eval_summery.txt")

    log_writter = open(output_eval_file, "a+")
    summery_writter = open(summery_log_file, "a+")

    # Eval!
    if args.screen_logs:
        print_log_to_screen("***** Running evaluation on {} *****".format(eval_type))
        print_log_to_screen("  Num examples = {} out of a total of {}".format(round(len(eval_dataset)*sampling_prob), len(eval_dataset)))
        print_log_to_screen("  Batch size = {}".format(args.eval_batch_size))

    eval_loss = 0.0
    nb_eval_steps = 0
    eval_accuracy = 0
    print_log_to_file(log_writter, "\n----- " + eval_type + " epoch: {}, global steps: {}-----\n".format(epoch,global_steps))
    print_log_to_file(summery_writter, "\n----- " + eval_type + " epoch: {}, global steps: {}-----\n".format(epoch,global_steps))
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        if random.random() > sampling_prob:
            continue
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            b_in_tensor = batch[0]
            b_out_tensor = batch[1]
            b_label_tensor = batch[2]
            if "bert_emb" in args.model_type:
                outputs = model(input_ids=b_in_tensor, lm_labels=b_label_tensor)
            elif "t5" in args.model_type:
                outputs = model(input_ids=b_in_tensor, decoder_lm_labels=b_label_tensor)
            else:
                outputs = model(b_in_tensor, b_out_tensor, decoder_lm_labels=b_label_tensor)
            tmp_eval_loss, logits = outputs[:2]
            batch_eval_loss = tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        out_label_ids = b_label_tensor.detach().cpu().numpy()
        input_seq_ids = b_in_tensor.detach().cpu().numpy()

        if 'emb' in args.model_type and acc_method != utils.after_upper_model_bleu_accuracy:
            pred_arr = utils.get_nearset_token(preds=preds, nbrs=nbrs)
        else:
            pred_arr = np.argmax(preds, axis=2)
        preds_tokens = []
        out_label_tokens = []
        input_tokens = []
        for i in range(len(pred_arr)):
            preds_tokens.append(tokenizer.decode(pred_arr[i], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            out_label_tokens.append(tokenizer.decode(out_label_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            input_tokens.append(tokenizer.decode(input_seq_ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            if args.screen_logs and args.log_examples and round(random.random()*100) % 10 == 0:
                print_log_to_screen("X:{}".format(tokenizer.decode(input_seq_ids[i][input_seq_ids[i] != 0])))
                print_log_to_screen("Y:{}".format(tokenizer.decode(out_label_ids[i][out_label_ids[i] != 0])))
                print_log_to_screen("PRED:{}\n".format(tokenizer.decode(pred_arr[i][(pred_arr[i] != 0) & (pred_arr[i] !=29193)])))

        if acc_method in [utils.simple_accuracy, utils.t5_exact]:
            batch_acc = acc_method(preds_tokens, out_label_tokens, do_logs=args.log_examples, log_writer=log_writter)
        elif acc_method == utils.after_upper_model_bleu_accuracy:
            batch_acc = acc_method(preds, out_label_tokens, tokenizer, upper_model, do_logs=args.log_examples, log_writer=log_writter)
        else: # bleu
            batch_acc = acc_method(preds_tokens, out_label_tokens, do_logs=args.log_examples, log_writer=log_writter, input_tokens=input_tokens)
        eval_accuracy += batch_acc
        eval_loss += batch_eval_loss
        print_log_to_file(log_writter,"\n----- "+eval_type+" epoch: {}, steps: {}, batch acc: {}, batch loss: {} -----\n".format(epoch, nb_eval_steps, batch_acc, batch_eval_loss))
        print_log_to_file(summery_writter,"\n" + eval_type + " epoch: {}, steps: {}, batch acc: {}, batch loss: {}\n".format(epoch,nb_eval_steps, batch_acc, batch_eval_loss))
        result = {"batch_eval_acc": batch_acc, "batch_eval_loss": eval_loss}
        results.update(result)

        summery_writter.flush()
        log_writter.flush()

    if nb_eval_steps > 0:
        eval_accuracy /= nb_eval_steps
        eval_loss /= nb_eval_steps


    log_msg = "\n----- " + eval_type + " epoch: {}, steps: {}, acc: {}, loss: {} -----\n".\
              format(epoch, nb_eval_steps, eval_accuracy, eval_loss)

    print_log_to_file(log_writter,log_msg)
    print_log_to_file(summery_writter,log_msg)

    summery_writter.close()
    log_writter.close()
    return {"eval_acc": eval_accuracy, "eval_loss": eval_loss}

