from transformers import BertForMaskedLM, BertTokenizer, T5Tokenizer, T5Model, BertModel
import torch
from scipy import spatial
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors


def embs2bert_sanity_check():
    masked_sent = "I am going for a walk in the [MASK]."
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertForMaskedLM.from_pretrained(model_name)
    input_ids = tokenizer.encode(masked_sent, return_tensors='pt')
    input_ids_output = bert(input_ids=input_ids)
    input_embeddings = bert.bert.embeddings.word_embeddings(input_ids)
    input_embeddings_output = bert(inputs_embeds=input_embeddings)
    assert torch.all(torch.eq(input_ids_output[0], input_embeddings_output[0]))
    print("embeddings passing validation experiment succeeded")


def get_1nn_sanity_check():
    model_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertForMaskedLM.from_pretrained(model_name)
    vocab = torch.tensor(list(range(30522)))
    print("starting building vocab embs")
    embs = bert.bert.embeddings.word_embeddings(vocab).detach()
    print("starting tree building")
    tree = spatial.KDTree(embs)
    print("starting knn building")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(embs)
    print("init random points")
    points = np.random.rand(100, 768)
    print("starting tree query")
    start = time.time()
    dists, neighbours = tree.query(points)
    end = time.time()
    print("kdtree took %f seconds" % (end - start))
    print("starting knn query")
    start = time.time()
    dists, neighbours = nbrs.kneighbors(points)
    end = time.time()
    print("nearest neighbours took %f seconds" % (end - start))
    assert tokenizer.convert_ids_to_tokens(neighbours) == ['face', 'face']
    print("knn validation experiment succeeded")


def t5_bert_closer_exp():
    masked_sent = "I am going for a walk in the [MASK]."
    bert_name = "lower_bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert = BertForMaskedLM.from_pretrained(bert_name)
    input_ids = tokenizer.encode(masked_sent, return_tensors='pt')
    last_hidden_states = bert.bert(input_ids=input_ids)[0][0].detach().numpy()
    t5_name = "t5-base"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
    t5 = T5Model.from_pretrained(t5_name)
    masked_sent = masked_sent.replace('[MASK]', '<extra_id_1>')
    input_ids = t5_tokenizer.encode(masked_sent, return_tensors='pt')
    t5_dec_last_hidden_states, t5_enc_last_hidden_states = t5(input_ids=input_ids)
    t5_dec_last_hidden_states, t5_enc_last_hidden_states = t5_dec_last_hidden_states[0].detach(), t5_enc_last_hidden_states[0].detach()
    vocab = torch.tensor(list(range(30522)))
    embs = bert.bert.embeddings.word_embeddings(vocab).detach()
    tree = spatial.KDTree(embs)
    dbert, nbert = tree.query(last_hidden_states)
    ddec, ndec = tree.query(t5_dec_last_hidden_states)
    denc, nenc = tree.query(t5_enc_last_hidden_states)
    print("original sentence: \"%s\"" % masked_sent)
    print("lower_bert sentence: \"%s\"" % ' '.join(tokenizer.convert_ids_to_tokens(nbert)))
    print("t5 decoder sentence: \"%s\"" % ' '.join(tokenizer.convert_ids_to_tokens(ndec)))
    print("t5 encoder sentence: \"%s\"" % ' '.join(tokenizer.convert_ids_to_tokens(nenc)))


if __name__ == "__main__":
    get_1nn_sanity_check()
    # t5_bert_closer_exp()