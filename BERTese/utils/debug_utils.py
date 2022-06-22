def set_debug_args(args):
    #args.logging_steps = 100
    args.train_batch_size = min(args.items_for_debug, 16)
    args.eval_batch_size = min(args.items_for_debug, 16)
    args.gradient_accumulation_steps = 1
    args.train_eval_sampling_prob = 1
    args.test_eval_sampling_prob = 1
    args.intermediate_examples_log_prob = 1
    args.dev_examples_log_prob = 1
    args.train_examples_log_prob = 1
    args.data_path = args.debug_data_path
    args.dont_save_model = True
    args.num_train_epochs = args.num_train_epochs if args.num_train_epochs > 1000 else min(args.num_train_epochs,1000)
    args.logging_sampling_prob = 1
    return args
