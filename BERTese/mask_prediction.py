from transformers.modeling_bert import BertForMaskedLM
from transformers import BertTokenizer
import torch

if not torch.cuda.is_available():
    from BERTese.bert_utils import MASK, MASK_DUMMY, CLS, CLS_DUMMY, SEP_DUMMY, SEP
else:
    from bert_utils import MASK, MASK_DUMMY, CLS, CLS_DUMMY, SEP_DUMMY, SEP


def init_vanilla_mask_prediction_model(model_name="bert-base-uncased", cache_dir=None):
    if cache_dir is not None:
        model = BertForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer


def predict_mask_token(model=None, tokenizer=None, sentence_a="", sentence_b="", cased=False, pred_count=10,
                       only_alphanumeric=False, no_cuda=False, cache_dir=None, add_cls_sep=True):
    if not (sentence_a + sentence_b).count(MASK) == 1:
        print("ERORR!" + sentence_a + sentence_b)
        raise ValueError("there should be exactly on MASK")

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    do_lower_case = not cased

    if do_lower_case:
        tokenizer.basic_tokenizer.do_lower_case = do_lower_case

    masked_a = sentence_a.replace(MASK, MASK_DUMMY).replace(CLS, CLS_DUMMY).replace(SEP, SEP_DUMMY)
    masked_b = sentence_b.replace(MASK, MASK_DUMMY).replace(CLS, CLS_DUMMY).replace(SEP, SEP_DUMMY)

    tokenized_text_a = tokenizer.tokenize(masked_a)
    if add_cls_sep:
        tokenized_text_a = [CLS] + tokenized_text_a + [SEP]
    tokenized_text_b = tokenizer.tokenize(masked_b)

    if len(tokenized_text_b) > 1 and add_cls_sep:
        tokenized_text_b = tokenized_text_b + [SEP]

    # tokenized_text = tokenizer.tokenize(text)
    tokenized_text = tokenized_text_a + tokenized_text_b

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = [i for i, x in enumerate(tokenized_text) if x == MASK_DUMMY][0]
    tokenized_text[masked_index] = MASK

    if CLS_DUMMY in tokenized_text:
        cls_index = [i for i, x in enumerate(tokenized_text) if x == CLS_DUMMY][0]
        tokenized_text[cls_index] = CLS

    if SEP_DUMMY in tokenized_text:
        sep_index = [i for i, x in enumerate(tokenized_text) if x == SEP_DUMMY][0]
        tokenized_text[sep_index] = SEP

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [1] * len(tokenized_text)
    for i in range(len(tokenized_text_a)):
        segments_ids[i] = 0

    # Load pre-trained model (weights)
    model.to(device)
    model.eval()

    # Convert inputs to PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = torch.tensor([segments_ids])
    segments_tensors = segments_tensors.to(device)

    # Predict all tokens
    predictions = model(tokens_tensor, segments_tensors)
    predictions = predictions[0]

    token_pred_count = pred_count - 1

    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    is_alphanumeric = str.isalpha(predicted_token[0])

    predicted_tokens = []
    while only_alphanumeric and not is_alphanumeric:
        predictions[0, masked_index, predicted_index] = -11100000
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        is_alphanumeric = str.isalpha(predicted_token[0])

    #print(predicted_token[0] + ", " + str(predicted_token_prob))
    predicted_tokens.append(predicted_token[0])

    while token_pred_count > 0:
        predictions[0, masked_index, predicted_index] = -11100000
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        if not only_alphanumeric or str.isalpha(predicted_token[0]):
            token_pred_count = token_pred_count - 1
            predicted_tokens.append(predicted_token[0])

    return predicted_tokens
