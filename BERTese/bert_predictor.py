from __future__ import absolute_import, division, print_function, unicode_literals
import torch

if not torch.cuda.is_available():
    from BERTese.models.models_embeddings_loss import BertModelEmbeddingLoss, T5WithLMHeadModelEmbeddingLoss
    from BERTese.models.t5_models import *
    from BERTese.models.modeling_model2model import Model2Model
    from BERTese.models.bertese_models import *
    from BERTese.training_utils import *
    from BERTese.mask_prediction import init_vanilla_mask_prediction_model
else:
    from models.models_embeddings_loss import BertModelEmbeddingLoss, T5WithLMHeadModelEmbeddingLoss
    from models.t5_models import *
    from models.modeling_model2model import Model2Model
    from models.bertese_models import *
    from training_utils import *
    from mask_prediction import init_vanilla_mask_prediction_model

# from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


def init_sanity_bertese_model(upper_model_name,  device):
    model = BerteseSanity.from_pretrained(
        upperbert_pretrained_model_name_or_path=upper_model_name,device=device)
    tokenizer = BertTokenizer.from_pretrained(upper_model_name)

    return model, tokenizer


def init_model_by_type(args, type, model_name="bert-base-uncased", cache_dir=None, bertese_lower_model_name="bert-base-uncased"):
    device,_ = get_device()
    type = type.lower()
    if type not in ["lama_bert", "bertese", "rewrite_bertese", "identity_seq2seq", "rewrite_seq2seq",
                    "lstm_rewrite_seq2seq", "lstm_identity_seq2seq",
                    "t5_rewrite_seq2seq", "t5_identity_seq2seq",
                    "t5_emb_rewrite_seq2seq", "t5_emb_identity_seq2seq",
                    "bert_emb_identity_seq2seq", "bert_emb_rewrite_seq2seq"]:
        raise ValueError("Invalid Type. select one of the values:" + str(type))
    if "seq2seq" in type:
        if "bert_emb" in type:
            model, tokenizer = init_bert_emb_model(model_name)
        elif "t5_emb" in type:
            model, tokenizer = init_t5_emb_loss_model(model_name)
        elif "t5" in type:
            model, tokenizer = init_t5_model(model_name)
        else:
            model, tokenizer = init_bert2bert_model(model_name, cache_dir)
    elif type == "lama_bert":
        model, tokenizer = init_vanilla_mask_prediction_model(model_name, cache_dir)
    elif "bertese" in type:
        if not args.sanity_model:
            model, tokenizer = init_bertese_model(bertese_lower_model_name, model_name, cache_dir, device, args.explicit_mask_loss_weight > 0,
                                      args.optimize_mask_softmin, args.tokens_dist_loss_weight > 0, args.do_dist_sum,args.vocab_gumble_softmax,
                                                  args.vocab_straight_through,args.mask_dis_met)
        else:
            model, tokenizer = init_sanity_bertese_model(model_name, device)

    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)

    return model, tokenizer


def init_bertese_model(bertese_lower_model_name, bertese_upper_model_name, cache_dir=None, device="cpu",
                       explicit_mask_loss= 0.0, optimize_mask_softmin=False, tokens_dist_loss =False,
                       do_dist_sum=False,vocab_gumble_softmax=False, vocab_straight_through=False,
                       mask_dis_met="l2_dis"):
    if cache_dir is not None:
        model = BerteseBertEmbeddingWithMask.from_pretrained(lowerbert_pretrained_model_name_or_path= bertese_lower_model_name,
                                                             upperbert_pretrained_model_name_or_path=bertese_upper_model_name,
                                                             device=device, cache_dir=cache_dir,
                                                             explicit_mask_loss=explicit_mask_loss,
                                                             optimize_mask_softmin=optimize_mask_softmin,
                                                             tokens_dist_loss=tokens_dist_loss,
                                                             do_dist_sum=do_dist_sum,
                                                             vocab_gumble_softmax=vocab_gumble_softmax,
                                                             vocab_straight_through=vocab_straight_through,
                                                             mask_dis_met=mask_dis_met)
        tokenizer = BertTokenizer.from_pretrained(bertese_upper_model_name, cache_dir=cache_dir)
    else:
        model = BerteseBertEmbeddingWithMask.from_pretrained(lowerbert_pretrained_model_name_or_path= bertese_lower_model_name,
                                                             upperbert_pretrained_model_name_or_path=bertese_upper_model_name,
                                                             device=device, explicit_mask_loss=explicit_mask_loss,
                                                             optimize_mask_softmin=optimize_mask_softmin,
                                                             tokens_dist_loss=tokens_dist_loss,
                                                             do_dist_sum=do_dist_sum,
                                                             vocab_gumble_softmax=vocab_gumble_softmax,
                                                             vocab_straight_through=vocab_straight_through,
                                                             mask_dis_met=mask_dis_met)
        tokenizer = BertTokenizer.from_pretrained(bertese_upper_model_name)
    return model, tokenizer


def init_bert_emb_model(model_name="bert-base-uncased", cache_dir=None):
    if cache_dir is not None:
        model = BertModelEmbeddingLoss.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        model = BertModelEmbeddingLoss.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model, tokenizer


"""
def init_bert2bert_emb_loss_model(model_name='lower_bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = PreTrainedEncoderDecoderEmbeddingLoss.from_pretrained(encoder_cls=BertModelEmbeddingLoss,
                                                                  decoder_cls=BertModelEmbeddingLoss,
                                                                  encoder_pretrained_model_name_or_path=model_name,
                                                                  decoder_pretrained_model_name_or_path=model_name)
    return model, tokenizer
"""

def init_bert2bert_model(model_name="lower_bert-large-uncased", cache_dir=None):
    if cache_dir is not None:
        model = Model2Model.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        model = Model2Model.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer
