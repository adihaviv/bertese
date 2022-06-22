from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration

try:
    from BERTese.models.models_embeddings_loss import BertModelEmbeddingLoss, T5WithLMHeadModelEmbeddingLoss, \
        PreTrainedEncoderDecoderEmbeddingLoss
except:
    from models.models_embeddings_loss import BertModelEmbeddingLoss, T5WithLMHeadModelEmbeddingLoss

T5_MASK = "<extra_id_1>"


def init_t5_model(model_name='t5-base'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def init_t5_emb_loss_model(model_name='t5-base'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5WithLMHeadModelEmbeddingLoss.from_pretrained(model_name)
    return model, tokenizer
