import os


def print_log_to_screen_and_file(writter, str, er=None):
    print_log_to_screen(str, er)
    print_log_to_file(writter, str, er)


def print_log_to_screen(str, er=None):
    try:
        print(str)
    except UnicodeEncodeError as e:
        tokens = str.split()
        for t in tokens:
            try:
                print(t , end=' ')
            except:
                print("token-unicode-error", end='')
        print("\n")
        if er is not None:
            print(er)


def print_log_to_file(writter, str, er=None, filter_unused=True):
    try:
        writter.write(str+"\n")
    except UnicodeEncodeError as e:
        tokens = str.split()
        for t in tokens:
            try:
                writter.write(t+" ")
            except:
                writter.write("token-unicode-error ")
        writter.write("\n")


def log_args(args, logs_pre_path):
    output_args_file = os.path.join(logs_pre_path, "args.txt")
    output_args_file = open(output_args_file, "w", 1)
    output_args_file.write(str(args.__dict__))
    output_args_file.flush()
    output_args_file.close()


def init_log_files(data_path, logs_pre_path):
    if not os.path.exists(logs_pre_path):
        os.makedirs(logs_pre_path, exist_ok=True)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path, logs_pre_path
