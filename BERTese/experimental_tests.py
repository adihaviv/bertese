from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from transformers import BertForMaskedLM, BertModel, BertTokenizer

if not torch.cuda.is_available():
    from BERTese.models.experimental_models import *
    import BERTese.training_utils as utils
else:
    from models.experimental_models import *
    import training_utils as utils


def bert_out_sanity_test():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    input_ids = tokenizer.encode("marcel alessandri died in [MASK].",
                                 return_tensors='pt',
                                 add_special_tokens=True)
    output = model(input_ids=input_ids)
    preds_distribution = output[0].detach().cpu().numpy()
    preds = np.argmax(preds_distribution, axis=2)
    output_tokens = tokenizer.decode(preds.flatten())
    print(output_tokens)


def bert_out_bert_in_token_emb_sanity_test():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    input_ids = tokenizer.encode("marcel alessandri died in [MASK].",
                                 return_tensors='pt',
                                 add_special_tokens=True)
    bert_output = model.bert(input_ids=input_ids)[0]
    output = model(inputs_embeds=bert_output)[0]
    preds_distribution = output.detach().cpu().numpy()[0]
    preds = np.argmax(preds_distribution, axis=1)
    output_tokens = tokenizer.decode(preds.flatten())
    print(output_tokens)


def bert_out_bert_in_emb_sanity_test():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLMWithEmeddings.from_pretrained(model_name)
    input_ids = tokenizer.encode("marcel alessandri died in [MASK].",
                                 return_tensors='pt',
                                 add_special_tokens=True)
    bert_output = model.bert(input_ids=input_ids)[0]
    output = model(inputs_embeds=bert_output)[0]
    preds_distribution = output.detach().cpu().numpy()[0]
    preds = np.argmax(preds_distribution, axis=1)
    output_tokens = tokenizer.decode(preds.flatten())
    print(output_tokens)


def bert_out_bert_in_emb_nn_sanity_test():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLMWithEmeddings.from_pretrained(model_name)
    vocab = torch.tensor(list(range(model.config.vocab_size)))
    nbrs = model.bert.embeddings.word_embeddings(vocab).detach().cpu().numpy()

    input_ids = tokenizer.encode("marcel alessandri died in [MASK].",
                                 return_tensors='pt',
                                 add_special_tokens=True)
    bert_output = model.bert(input_ids=input_ids)[0]
    bert_output_np = bert_output.detach().cpu().numpy()
    bert_nn = utils.get_nearset_token(preds=bert_output_np, nbrs=nbrs)
    bert_nn_tokens = tokenizer.decode(bert_nn.flatten())
    print(bert_nn_tokens)



def bert_out_bert_in_emb_nn_sanity_test():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLMWithEmeddings.from_pretrained(model_name)
    vocab = torch.tensor(list(range(model.config.vocab_size)))
    nbrs = model.bert.embeddings.word_embeddings(vocab).detach().cpu().numpy()

    input_ids = tokenizer.encode("marcel alessandri died in [MASK].",
                                 return_tensors='pt',
                                 add_special_tokens=True)
    bert_output = model.bert(input_ids=input_ids)[0]
    bert_output_np = bert_output.detach().cpu().numpy()
    bert_nn = utils.get_nearset_token(preds=bert_output_np, nbrs=nbrs)
    bert_nn_tokens = tokenizer.decode(bert_nn.flatten())
    print(bert_nn_tokens)


def bert_out_bert_in_nn_sanity_test():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    vocab = torch.tensor(list(range(model.config.vocab_size)))
    nbrs = model.bert.embeddings.word_embeddings(vocab).detach().cpu().numpy()

    input_ids = tokenizer.encode("marcel alessandri died in [MASK].",
                                 return_tensors='pt',
                                 add_special_tokens=True)
    bert_output = model.bert(inputs_embeds=input_ids)[0]
    bert_output_np = bert_output.detach().cpu().numpy()
    bert_nn = utils.get_nearset_token(preds=bert_output_np, nbrs=nbrs)
    bert_nn_tokens = tokenizer.decode(bert_nn.flatten())
    print(bert_nn_tokens)


def sanity_bert():
    import BERTese.mask_prediction as m
    import torch

    bert_base_model, base_tokenizer = m.init_vanilla_mask_prediction_model(model_name="bert-base-uncased")
    bert_large_model, large_tokenizer = m.init_vanilla_mask_prediction_model(model_name="bert-large-uncased")

    input = "ira allen was born in [MASK]."
    a = torch.tensor(base_tokenizer.encode(input))
    a = bert_base_model(input_ids=torch.tensor([base_tokenizer.encode(input)]))[0].detach().cpu().numpy()
    print(base_tokenizer.decode(np.argmax(a,axis=2)[0]))

if __name__ == '__main__':
    sanity_bert()
    """
    print("--- EXP: lower_bert -> lm head ---\n")
    bert_out_sanity_test()
    print("--- EXP: lower_bert -> lower_bert input emb -> lm head ---\n")
    bert_out_bert_in_token_emb_sanity_test()
    print("--- EXP: lower_bert -> lower_bert replace embedding layer-> lm head ---\n")
    bert_out_bert_in_emb_sanity_test()
    print("--- EXP: lower_bert -> lower_bert replace embedding layer-> NN -> lm head ---\n")
    bert_out_bert_in_emb_nn_sanity_test()
    """
