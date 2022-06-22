from __future__ import absolute_import, division, print_function, unicode_literals

from torch import nn
import torch.nn.functional as F
from torch.nn import Softmin, CrossEntropyLoss, MSELoss
import torch
from transformers import BertForMaskedLM, BertTokenizer, BertConfig, BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings
import numpy as np
import faiss
import random


if not torch.cuda.is_available():
    import BERTese.models.modeling_unsupervised_model2model as m
    from BERTese.models.models_embeddings_loss import BertModelEmbeddingLoss
else:
    try:
        import modeling_unsupervised_model2model as m
        from models_embeddings_loss import BertModelEmbeddingLoss
    except:
        import models.modeling_unsupervised_model2model as m
        from models.models_embeddings_loss import BertModelEmbeddingLoss

REMOVE_DROPOUT = False

class BerteseBertEmbeddingWithMask(BertPreTrainedModel):
    def __init__(self, config, upper_bert, lower_bert, tokenizer, device, explicit_mask_loss=False,
                 optimize_softmin=False, tokens_dist_loss=False, do_dist_sum = False, vocab_gumble_softmax=False,
                 vocab_straight_through = False, mask_dis_met = "l2_dis"):
        #todo: remove this - its only for debug!
        if REMOVE_DROPOUT:
            config.attention_probs_dropout_prob = 0
            config.hidden_dropout_prob = 0
        super().__init__(config)
        self.tokenizer = tokenizer
        self.lower_bert = lower_bert
        #self.lower_bert.config.attention_probs_dropout_prob = 0
        #self.lower_bert.config.hidden_dropout_prob = 0

        self.upper_bert = upper_bert
        self.upper_bert.requires_grad = False
        for param in self.upper_bert.parameters():
            param.requires_grad = False
        #for param in self.lower_bert.bert.embeddings.parameters():
        #    param.requires_grad = False

        self.model_device = device
        self.one_tensor = torch.tensor(1.0).to(self.model_device)
        self.zero_tensor = torch.tensor(0.0).to(self.model_device)
        self.softmin = Softmin(dim=1)
        self.optimize_softmin = optimize_softmin
        self.explicit_mask_loss = explicit_mask_loss
        self.tokens_dist_loss = tokens_dist_loss
        self.do_dist_sum = do_dist_sum
        self.vocab_gumble_softmax = vocab_gumble_softmax
        self.vocab_straight_through = vocab_straight_through
        self.mask_dis_met = mask_dis_met

        vocab = torch.tensor(list(range(self.upper_bert.config.vocab_size)),dtype=torch.long)
        vocab_emb = self.get_bert_embeddings(vocab)
        self.vocab_emb = vocab_emb.to(device)
        self.vocab_ids = vocab.to(device)

        #self.bert_mask_embedding = self.get_bert_embeddings(
        #    torch.tensor(self.tokenizer.encode("[MASK]", add_special_tokens=False))).to(self.model_device)
        #self.bert_sep_embedding = self.get_bert_embeddings(
        #    torch.tensor(self.tokenizer.encode("[SEP]", add_special_tokens=False))).to(self.model_device)
        self.bert_mask_embedding = upper_bert.bert(
            torch.tensor([tokenizer.encode("[MASK]", add_special_tokens=False)]))[0].flatten().to(self.model_device)
        self.bert_sep_embedding = upper_bert.bert(
           torch.tensor([tokenizer.encode("[SEP]", add_special_tokens=False)]))[0].flatten().to(self.model_device)

    def get_bert_embeddings(self, tokens):
        return self.upper_bert.bert.embeddings.word_embeddings(tokens)

    @classmethod
    def from_pretrained(
        cls,
        lowerbert_pretrained_model_name_or_path=None,
        upperbert_pretrained_model_name_or_path=None,
        device="cpu",
        explicit_mask_loss=False,
        optimize_mask_softmin=False,
        tokens_dist_loss=False,
        do_dist_sum=False,
        vocab_gumble_softmax=False,
        vocab_straight_through=False,
        mask_dis_met = "l2_dis",
        *model_args,
        **kwargs
    ):
        config = kwargs.pop("config", None)
        if not isinstance(config, BertConfig):
            config = BertConfig.from_pretrained(upperbert_pretrained_model_name_or_path, **kwargs)
            #kwargs["config"] = config
            #config.attention_probs_dropout_prob = 0
            #config.hidden_dropout_prob = 0

        tokenizer = BertTokenizer.from_pretrained(upperbert_pretrained_model_name_or_path, *model_args, **kwargs)
        upper_bert = BertForMaskedLM.from_pretrained(upperbert_pretrained_model_name_or_path, *model_args, **kwargs)
        lower_bert = BertModelEmbeddingLoss.from_pretrained(lowerbert_pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(config, upper_bert, lower_bert, tokenizer, device, explicit_mask_loss,
                    optimize_mask_softmin, tokens_dist_loss, do_dist_sum, vocab_gumble_softmax,
                    vocab_straight_through,mask_dis_met)

        return model

    @classmethod
    def from_pretrained_models(
            cls,
            upper_bert,
            lower_bert,
            tokenizer,
            device="cpu",
            explicit_mask_loss = False,
            optimize_softmin = False,
            tokens_dist_loss=False,
            do_dist_sum = False,
            **kwargs
    ):
        config = kwargs.pop("config", None)
        if not isinstance(config, BertConfig):
            config = upper_bert.config
            #kwargs["config"] = config
            #config.attention_probs_dropout_prob = 0
            #config.hidden_dropout_prob = 0

        model = cls(config, upper_bert, lower_bert, tokenizer, device, explicit_mask_loss,
                    optimize_softmin, tokens_dist_loss, do_dist_sum)

        return model

    def forward(
            self,
            input_ids=None,
            mask_label=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            disable_embeddings_dropout=False,
            temperature = 1e-5
    ):

        self.upper_bert.eval()
        #self.lower_bert.bert.embeddings.eval()
        lower_bert_outputs = self.lower_bert(input_ids)#, disable_embeddings_dropout=disable_embeddings_dropout)
        lower_bert_lhs_real = lower_bert_outputs[0]

        if self.vocab_gumble_softmax or self.vocab_straight_through:
            all_dist = -1 * _compte_all_dist(lower_bert_lhs_real, self.vocab_emb)
            if self.vocab_gumble_softmax:
                all_dist_tau = 1e-3
                phi = torch.log(torch.softmax(all_dist/all_dist_tau, dim=2))
                all_dist_st = torch.stack([F.gumbel_softmax(phi[b], tau=temperature, hard=True) for b in range(all_dist.shape[0])])

            elif self.vocab_straight_through:
                all_dist_tau = torch.torch.FloatTensor(1.0).to(lower_bert_lhs_real.device)
                y_soft = torch.softmax(all_dist / all_dist_tau,dim=2)

                # Straight through.
                index = torch.max(y_soft,dim=2,keepdim=True)[1]
                y_hard = torch.zeros_like(all_dist).scatter_(2, index, 1.0)
                all_dist_st = y_hard - y_soft.detach() + y_soft

            lower_bert_lhs = torch.matmul(all_dist_st, self.vocab_emb)  # bmm(lh_gs, self.vocab_emb.unsqueeze(0))
        else:
            lower_bert_lhs = lower_bert_lhs_real

        upper_bert_logits = self.upper_bert(inputs_embeds=lower_bert_lhs)[0]
        outputs = (lower_bert_lhs, upper_bert_logits) + lower_bert_outputs[2:]

        if self.mask_dis_met == "l2_dist":
            mask_dist_mat = torch.norm(lower_bert_lhs - self.bert_mask_embedding, dim=2)
        else:
            cos = nn.CosineSimilarity(dim=0)
            mask_dist_mat = []
            for batch_i in range(lower_bert_lhs.shape[0]):
                seq_dist = []
                for seq_j in range(lower_bert_lhs.shape[1]):
                    similarity = cos(lower_bert_lhs[batch_i][seq_j], self.bert_mask_embedding)
                    seq_dist.append(similarity)
                mask_dist_mat.append(torch.stack(seq_dist,dim=0).to(lower_bert_lhs.device))
            mask_dist_mat = torch.stack(mask_dist_mat,dim=0).to(lower_bert_lhs.device)
            mask_dist_mat /= np.sqrt(self.bert_mask_embedding.shape[0])
        p = torch.softmax(mask_dist_mat, dim=1)

        #p = self.softmin(mask_dist_mat)
        #print(p)
        #print(torch.argmax(p,dim=1))

        v_tag = self.upper_bert.bert(inputs_embeds=lower_bert_lhs)[0]
        mask_pred = torch.bmm(p.unsqueeze(1), v_tag).squeeze()

        label_logits = self.upper_bert.cls(mask_pred)
        outputs = (label_logits,) + outputs
        outputs = (mask_dist_mat,) + outputs
        outputs = (p,) + outputs

        explicit_mask_loss = torch.tensor(0.0).to(p.device)
        if self.explicit_mask_loss:
            loss_func = MSELoss()
            if self.optimize_softmin:
                explicit_mask_loss = loss_func(torch.max(p, dim=1)[0], self.one_tensor)
            else:
                explicit_mask_loss = loss_func(torch.min(mask_dist_mat, dim=1)[0], self.zero_tensor)

        outputs = (explicit_mask_loss,) + outputs

        loss_func = MSELoss()
        sep_dist_mat = torch.norm(lower_bert_lhs - self.bert_sep_embedding, dim=2)
        sep_loss = loss_func(torch.min(sep_dist_mat, dim=1)[0], self.zero_tensor)
        outputs = (sep_loss,) + outputs

        loss_func = MSELoss()
        #all_dist = _compte_all_dist_dot_product(lower_bert_lhs, self.vocab_emb)
        all_dist = _compte_all_dist(lower_bert_lhs, self.vocab_emb)

        if self.do_dist_sum:
            vocab_loss = loss_func(torch.sum(torch.min(all_dist,dim=2, keepdim=True)[0]), self.zero_tensor)
        else:
            vocab_loss = loss_func(torch.mean(torch.min(all_dist, dim=2)[0]), self.zero_tensor)

        #print(torch.min(all_dist, dim=2, keepdim=True))
        outputs = (vocab_loss,) + outputs

        if mask_label is not None:
            loss_func = CrossEntropyLoss()
            loss = loss_func(label_logits.view(-1, self.config.vocab_size), mask_label.view(-1))
            outputs = (loss,) + outputs

        return outputs #[loss], vocab_loss, sep_loss, mask_dist_softmin, explicit_mask_loss, label_logits, lower_bert_lhs, upper_bert_logits


def _compte_all_dist(a,b):
    """
    a_norm = a.norm(dim=2)[:, :, None]
    b_t = b.permute(0, 2, 1).contiguous()
    b_norm = b.norm(dim=2)[:, None]
    all_dist = torch.sqrt(torch.sum(a_norm + b_norm - 2.0 * torch.bmm(a, b_t)))
    """
    A = a
    B = b
    if len(a.shape) == 3:
        A = a.view(a.shape[0]*a.shape[1],a.shape[2])
    if len(b.shape) == 3:
        B = b.view(b.shape[0]*b.shape[1],b.shape[2])

    sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
    sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
    #
    #res = torch.sqrt(sqrA - 2 * torch.mm(A, B.t()) + sqrB)
    res = sqrA - 2 * torch.mm(A, B.t()) + sqrB
    if len(a.shape) == 3:
        res = res.view(a.shape[0],a.shape[1],res.shape[1])
    return res


def _compte_all_dist_dot_product(a,b):
    A = a
    B = b
    if len(a.shape) == 3:
        A = a.view(a.shape[0] * a.shape[1], a.shape[2])
    if len(b.shape) == 3:
        B = b.view(b.shape[0] * b.shape[1], b.shape[2])


    res = torch.mm(A, B.t())
    if len(a.shape) == 3:
        res = res.view(a.shape[0], a.shape[1], res.shape[1])
    return res


class BerteseSanity(BertPreTrainedModel):

    def __init__(self, config, upper_bert, lower_ff, tokenizer, device):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.upper_bert = upper_bert
        self.upper_bert.requires_grad = False
        for param in self.upper_bert.parameters():
            param.requires_grad = False
        self.lower_ff = lower_ff

        self.model_device = device
        self.zero_tensor = torch.tensor(0.0).to(self.model_device)

        vocab = torch.tensor(list(range(self.upper_bert.config.vocab_size)))
        vocab_emb = self.get_bert_embeddings(vocab)
        self.vocab_emb = vocab_emb.view(vocab_emb.shape[0], 1, -1).to(device)

    def get_bert_embeddings(self, tokens):
        return self.upper_bert.bert.embeddings.word_embeddings(tokens)

    @classmethod
    def from_pretrained(
        cls,
        upperbert_pretrained_model_name_or_path=None,
        device="cpu",
        *model_args,
        **kwargs
    ):
        config = kwargs.pop("config", None)
        if not isinstance(config, BertConfig):
            config = BertConfig.from_pretrained(upperbert_pretrained_model_name_or_path, **kwargs)

        tokenizer = BertTokenizer.from_pretrained(upperbert_pretrained_model_name_or_path, *model_args, **kwargs)
        upper_bert = BertForMaskedLM.from_pretrained(upperbert_pretrained_model_name_or_path, *model_args, **kwargs)
        lower_ff = nn.Linear(768, 768)
        model = cls(config, upper_bert,lower_ff, tokenizer, device)

        return model

    def forward(
            self,
            input_ids=None,
            mask_label=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):

        self.upper_bert.eval()
        lower_ff_outputs = self.lower_ff(input_ids)
        upper_bert_logits = self.upper_bert(inputs_embeds=lower_ff_outputs)[0]

        vocab_loss = torch.tensor(0.0)
        # make sure that the vectors are close to the real bert tokens
        if self.tokens_dist_loss:
            loss_func = MSELoss()
            a = torch.autograd.Variable(lower_ff_outputs.view(lower_ff_outputs.shape[0] * lower_ff2_outputs.shape[1],lower_ff2_outputs.shape[2]),
                                    requires_grad=lower_ff_outputs.requires_grad).unsqueeze(0)
            #todo: check the required grad here
            b = torch.autograd.Variable(self.vocab_emb.view(self.vocab_emb.shape[0],self.vocab_emb.shape[2]),
                                    requires_grad=lower_ff_outputs.requires_grad).unsqueeze(0)

            all_dist =_compte_all_dist_dot_product(a,b)

            #all_dist[all_dist != all_dist] = 0  # replace nan values with 0
            #all_dist = torch.clamp(all_dist, 0.0, np.inf)

            if self.do_dist_sum:
                vocab_loss = loss_func(torch.sum(torch.min(all_dist,dim=2, keepdim=True)[0]),self.zero_tensor)
            else:
                vocab_loss = torch.mean(torch.min(all_dist, dim=2, keepdim=True)[0],2)
            print(torch.min(all_dist, dim=1, keepdim=True)[0])
        outputs = (vocab_loss,)

        return outputs #[loss], vocab_loss, sep_loss, mask_dist_softmin, explicit_mask_loss, label_logits, lower_bert_lhs, upper_bert_logit


def _get_nearset_token(preds, nbrs):
    index = faiss.IndexFlatL2(nbrs.shape[-1])
    index.add(nbrs)
    D, I = index.search(preds.reshape(-1, preds.shape[2]), 1)
    return I.reshape(preds.shape[0], -1)

"""
def bertese_sanity_test():

    with torch.no_grad():
        lower_model_name = "/specific/netapp5_2/gamir/advml19/yuvalk/project/BERTese/output/bert_emb_identity_seq2seq_19_05_19_49_52/bert-base-uncased-identity-mse-sum-checkpoint-51031/"
        upper_model_name = "bert-base-uncased"
        max_length = 80
        input_sentence = "the native language of louis - jean - marie daubenton is [MASK]."
        tokenizer = BertTokenizer.from_pretrained(upper_model_name)
        lower_model = BertModelEmbeddingLoss.from_pretrained(lower_model_name)
        upper_model = BertForMaskedLM.from_pretrained(upper_model_name)

        vocab = torch.tensor(list(range(upper_model.config.vocab_size)))
        nbrs = upper_model.bert.embeddings.word_embeddings(vocab)
        nbrs = nbrs.detach().numpy()

        input_ids = tokenizer.encode(input_sentence,
                                     pad_to_max_length=True,
                                     max_length=max_length,
                                     return_tensors='pt')
        out_embs = lower_model(input_ids=input_ids)[0]
        out_embs_numpy = out_embs.detach().numpy()

        lower_output_ids = _get_nearset_token(out_embs_numpy, nbrs)[0]
        lower_output = tokenizer.decode(lower_output_ids)

        cleaned_lower_output_nn = lower_output[lower_output.index(' '): lower_output.index('[SEP]')].strip()
        print("lower model NN output is: '%s'" % cleaned_lower_output_nn)
        assert cleaned_lower_output_nn == input_sentence

        output = upper_model(inputs_embeds=out_embs)[0].detach().numpy()
        predicted = np.argmax(output[0], axis=1)
        print("upper model output with lower output as input is: '%s'" % tokenizer.decode(predicted))

        model_2 = BerteseBertEmbeddingWithMask.from_pretrained_models(upper_bert=upper_model, lower_bert=lower_model,
                                                                      tokenizer=tokenizer, mask_token_loss=False)
        label_logits, lower_models_last_hs, upper_model_logits = model_2(input_ids=input_ids)

        out_label_embs_numpy = label_logits.detach().numpy()
        print("predicted token:", tokenizer.decode([np.argmax(out_label_embs_numpy)]))

        rewrite_embs_numpy = lower_models_last_hs.detach().numpy()
        lower_output_ids = _get_nearset_token(rewrite_embs_numpy, nbrs)[0]

        print("rewrite NN:", tokenizer.decode(lower_output_ids))

    # cls for rewrite?
"""

def bert_sanity_test():
    upper_model_name = "bert-base-uncased"
    max_length = 80
    input_sentence = "the native language of louis - jean - marie daubenton is [MASK]."
    tokenizer = BertTokenizer.from_pretrained(upper_model_name)
    model = BertForMaskedLM.from_pretrained(upper_model_name)

def dist_test():
    A = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[0.7,0.8,0.9]])
    B = torch.tensor([[1.0,2.0,3.0]])


    print(_compte_all_dist(A,B))

if __name__ == '__main__':
    dist_test()
