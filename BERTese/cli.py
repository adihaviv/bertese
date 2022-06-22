import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lama_data_path",
                        default=r"/specific/netapp5_2/gamir/adi/Bertese/data/",
                        type=str,
                        required=False,
                        help="The original BERTese input data files")
    parser.add_argument("--data_path",
                        default=r"/specific/netapp5_2/gamir/adi/git/BERTese/BERTese/seq2seq/data/",
                        type=str,
                        required=False,
                        help="The input data files")
    parser.add_argument("--debug_data_path",
                        default=r"/specific/netapp5_2/gamir/adi/git/BERTese/BERTese/sanity-test-bertese/data/",
                        type=str,
                        required=False,
                        help="The input data files")
    parser.add_argument("--model_name",
                        default="/specific/netapp5_2/gamir/adi/BERT_models/bertese_lower_bert_pre_train_models/bert-base-uncased-identity-mse-sum-checkpoint-51031/",
                        type=str,
                        required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese")
    parser.add_argument("--upper_model_name",
                        default="bert-base-uncased",
                        type=str,
                        required=False,
                        help="Bert pre-trained model selected in the list: lower_bert-base-uncased, "
                             "bert-large-uncased, lower_bert-base-cased, lower_bert-base-multilingual, lower_bert-base-chinese")
    parser.add_argument("--mask_predict_model_name",
                        default="bert-base-uncased",
                        type=str,
                        required=False,
                        help="Bert pre-trained model selected in the list: lower_bert-base-uncased, "
                             "lower_bert-large-uncased, lower_bert-base-cased, lower_bert-base-multilingual, lower_bert-base-chinese")
    parser.add_argument("--output_dir",
                        default="/specific/netapp5_2/gamir/adi/git/BERTese/BERTese/seq2seq/results",
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=15,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--dont_save_model",
                        default=False,
                        action='store_true',
                        help="Whether to save the model.")
    parser.add_argument("--do_eval_dev",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--split_relations",
                        default=False,
                        action='store_true',
                        help="Whether to split the dataset per relations or flat everythin.")
    parser.add_argument("--debug",
                        default=False,
                        action='store_true',
                        help="changes the training eval sampling to 1.0 and take only 10 of the examples")
    parser.add_argument("--override_nee",
                        default=False,
                        action='store_true',
                        help="recreate nee files")
    parser.add_argument("--verbose_e",
                        default=False,
                        action='store_true',
                        help="changes the training eval sampling to 1.0 and take only 10 of the examples")
    parser.add_argument("--verbose",
                        default=False,
                        action='store_true',
                        help="changes the training eval sampling to 1.0 and take only 10 of the examples")
    parser.add_argument("--stop_on_e",
                        default=False,
                        action='store_true',
                        help="changes the training eval sampling to 1.0 and take only 10 of the examples")
    parser.add_argument("--do_eval_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--tokens_dist_lr",
                        #default=5e+5,
                        default=0,
                        type=float,
                        help="The initial learning rate for the tokens dist loss.")
    parser.add_argument("--tokens_dist_scale",
                        default=1,
                        type=float,
                        help="The initial learning rate for the tokens dist loss.")
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='weight decay')
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float,
                        default=1,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--override_output',
                        default=True,
                        action='store_true',
                        help="override the output directory")
    parser.add_argument('--filter_identity',
                        default=False,
                        action='store_true',
                        help="if true we filter the BERTese data to the items lower_bert is correct already")
    parser.add_argument('--override_data_split',
                        default=False,
                        action='store_true',
                        help="override the output directory")
    parser.add_argument('--random_input',
                        default=False,
                        action='store_true',
                        help="created a random input in the size of max len")
    parser.add_argument('--dont_include_special_tokens',
                        default=False,
                        action='store_true',
                        help="created a random input in the size of max len")
    parser.add_argument('--sgd',
                        default=False,
                        action='store_true',
                        help="created a random input in the size of max len")
    parser.add_argument('--override_to_one_token',
                        default=None,
                        type=str,
                        required=False,
                        help="override the output directory")
    parser.add_argument('--train_eval_sampling_prob',
                        type=float,
                        default=0.1,
                        help='Percentage of training samples to evaluate training accuracy on (at the end of each epoch)')
    parser.add_argument('--explicit_mask_loss_weight',
                        type=float,
                        default=0.0,
                        help='the weight we give the explicit mask loss comparing to the label loss')
    parser.add_argument('--sep_loss_weight',
                        type=float,
                        default=0.0,
                        help='the weight we give the explicit mask loss comparing to the label loss')
    parser.add_argument('--logging_sampling_prob',
                        type=float,
                        default=0.1,
                        help='Percentage of training samples to evaluate training accuracy on (at the end of each epoch)')
    parser.add_argument('--test_eval_sampling_prob',
                        type=float,
                        default=1.0,
                        help='Percentage of training samples to evaluate training accuracy on (at the end of each epoch)')
    parser.add_argument('--intermediate_examples_log_prob',
                        type=float,
                        default=0.0,
                        help='Percentage of samples to log after every epoch')
    parser.add_argument('--train_examples_log_prob',
                        type=float,
                        default=0.1,
                        help='Percentage of samples to log after every epoch')
    parser.add_argument('--dev_examples_log_prob',
                        type=float,
                        default=1.0,
                        help='Percentage of samples to log after every epoch')
    parser.add_argument('--dummy',
                        default="False",
                        action='store_true',
                        required=False,
                        help="for testing things")
    parser.add_argument('--evaluate_during_training',
                        default=False,
                        action='store_true',
                        required=False,
                        help="for testing things")
    parser.add_argument('--logging_steps',
                        type=int,
                        default=3955,
                        help="amount of steps to run eval and log")
    parser.add_argument('--max_steps',
                        type=int,
                        default=0,
                        help="")
    parser.add_argument('--fp16_opt_level',
                        type=str,
                        default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--items_for_debug",
                        default=100,
                        type=int,
                        help="the amount of items we going to debug on")
    parser.add_argument('--save_steps',
                        type=int,
                        default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--warmup_steps",
                        default=100,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--debug_uuid",
                        default="",
                        type=str,
                        required=False,
                        help="test only the provided uuid")

    parser.add_argument("--acc_method",
                        default="bleu",
                        type=str,
                        required=False,
                        help="exact, bleu, t5_exact")

    parser.add_argument("--split_method",
                        default="random_split",
                        type=str,
                        required=False,
                        help="random_split, relation_split")

    #Todo: change model_type and subject_replace to model type, experiment? more clear, its getting messy
    parser.add_argument("--subject_replace",
                        default="none",
                        type=str,
                        required=False,
                        help="none, constant, relation")
    parser.add_argument("--model_type",
                        default="bertese",
                        type=str,
                        required=False,
                        help="bertese, bertese_rewrite, identity_seq2seq, rewrite_seq2seq, lstm_rewrite_seq2seq, lstm_idntity_seq2seq")
    parser.add_argument('--log_examples',
                        default=False,
                        action='store_true',
                        required=False,
                        help="for testing things")
    parser.add_argument('--manual_test',
                        default=False,
                        action='store_true',
                        required=False,
                        help="for testing things")
    parser.add_argument('--override_data_dump',
                        default=False,
                        action='store_true',
                        required=False,
                        help="for testing things")
    parser.add_argument('--screen_logs',
                        default=False,
                        action='store_true',
                        required=False,
                        help="for testing things")
    parser.add_argument('--log_tensorx',
                        default=False,
                        action='store_true',
                        required=False,
                        help="log to tensorboardX")
    parser.add_argument("--intermediate_screen_logs",
                        default=False,
                        action='store_true',
                        help="log to screen intermediate results")
    parser.add_argument("--no_intermediate_evaluation",
                        default=False,
                        action='store_true',
                        help="log to screen intermediate results")
    parser.add_argument("--bert_pred_sanity_test",
                        default=False,
                        action='store_true',
                        help="assert that the bert upper model return the same results as a vanilla bert")
    parser.add_argument("--sanity_model",
                        default=False,
                        action='store_true',
                        help="uses the bertese sanity model")
    parser.add_argument("--vocab_gumble_softmax",
                        default=False,
                        action='store_true',
                        help="if set to true uses gamble softmax to get replace the lower bert embeddings with the "
                             "nearset vocabelry embedding ")
    parser.add_argument("--vocab_straight_through",
                        default=False,
                        action='store_true',
                        help="if set to true uses gamble softmax to get replace the lower bert embeddings with the "
                             "nearset vocabelry embedding ")
    parser.add_argument("--debug_single_item_index",
                        default=-1, type=int)
    parser.add_argument("--disable_embeddings_dropout",
                        default=False,
                        action='store_true',
                        help="if true we optimize to have the normalized distance minimized using the softmin")
    parser.add_argument("--optimize_mask_softmin",
                        default=False,
                        action='store_true',
                        help="if true we optimize to have the normalized distance minimized using the softmin")
    parser.add_argument('--lpaqa',
                        default=False,
                        action='store_true',
                        required=False,
                        help="if set to true we use the LPAQA settings. switching to thier training set etc. ")
    parser.add_argument('--do_dist_sum',
                        default=False,
                        action='store_true',
                        required=False,
                        help="if set to true we use the LPAQA settings. switching to thier training set etc. ")
    parser.add_argument('--debug_identity_examples',
                        default=False,
                        action='store_true',
                        required=False,
                        help="if set to true we use the LPAQA settings. switching to thier training set etc. ")
    parser.add_argument('--tokens_dist_loss_weight',
                        type=float,
                        default=0.0,
                        help='the weight we give the tokens_dist_loss')
    parser.add_argument('--label_loss_weight',
                        type=float,
                        default=1.0,
                        help='the weight we give the tokens_dist_loss')
    parser.add_argument('--mask_dis_met',
                        choices=["cosine_sim", "l2_dis"],
                        default="l2_norm",
                        help='the weight we give the tokens_dist_loss')

    return parser.parse_args()