from torch.nn import MSELoss, CosineSimilarity
import torch
from transformers import BertForMaskedLM, BertTokenizer, PreTrainedEncoderDecoder,  T5Tokenizer
from transformers.modeling_t5 import T5ForConditionalGeneration
import numpy as np
import faiss

"""
if not torch.cuda.is_available():
    from BERTese.models.dropout_configurable_bert_models import DropoutEmbeddingsConfiguredBertForMaskedLM
else:
    try:
        from dropout_configurable_bert_models import BertModelEmbeddingLoss
    except:
        from models.dropout_configurable_bert_models import DropoutEmbeddingsConfiguredBertForMaskedLM
"""

class T5WithLMHeadModelEmbeddingLoss(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        for param in self.bert.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    def forward(self, **kwargs):
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        lm_labels = kwargs.pop("decoder_lm_labels", None)

        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
              # Convert encoder inputs in embeddings if needed
            hidden_states = kwargs_encoder.pop("inputs_embeds", None)
            if hidden_states is None:
               encoder_inputs_ids = kwargs_encoder.pop("input_ids")
               hidden_states = self.shared(encoder_inputs_ids)  # Convert inputs in embeddings

            encoder_outputs = self.encoder(hidden_states, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_outputs = ()

        # Decode
        # Convert decoder inputs in embeddings if needed
        hidden_states = kwargs_decoder.pop("inputs_embeds", None)
        if hidden_states is None:
            decoder_inputs_ids = kwargs_decoder.pop("input_ids")
            hidden_states = self.shared(decoder_inputs_ids)

        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get("attention_mask", None)
        decoder_outputs = self.decoder(hidden_states, **kwargs_decoder)

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)

        decoder_outputs = (sequence_output,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            lm_labels_embeddings = self.bert.bert.embeddings.word_embeddings(lm_labels)
            shift_logits = sequence_output[..., :-1, :].contiguous()
            shift_labels = lm_labels_embeddings[..., 1:, :].contiguous()
            loss_fct = MSELoss()
            loss = loss_fct(shift_logits.view(-1), shift_labels.view(-1))
            decoder_outputs = (
                loss,
            ) + decoder_outputs
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        return decoder_outputs + encoder_outputs


class BertModelEmbeddingLoss(BertForMaskedLM):
    def forward(
            self,
            input_ids=None,
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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]

        outputs = (sequence_output,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            lm_labels_embeddings = self.bert.embeddings.word_embeddings(masked_lm_labels)
            shift_logits = sequence_output[..., :-1, :].contiguous()
            shift_labels = lm_labels_embeddings[..., 1:, :].contiguous()
            loss_fct = CosineSimilarity
            ltr_lm_loss = loss_fct(shift_logits.view(-1), shift_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        if lm_labels is not None:
            lm_labels_embeddings = self.bert.embeddings.word_embeddings(lm_labels)
            # max_fct = CosineSimilarity(dim=1, eps=1e-6)
            # ltr_lm_loss = - torch.mean(max_fct(sequence_output.squeeze(), lm_labels_embeddings.squeeze()))
            loss_fct = MSELoss(reduction='sum')
            ltr_lm_loss = loss_fct(sequence_output.squeeze(), lm_labels_embeddings.squeeze())
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


"""
class BertModelEmbeddingLoss(DropoutEmbeddingsConfiguredBertForMaskedLM):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            disable_embeddings_dropout = False
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            disable_embeddings_dropout = disable_embeddings_dropout
        )

        sequence_output = outputs[0]

        outputs = (sequence_output,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            lm_labels_embeddings = self.bert.embeddings.word_embeddings(masked_lm_labels)
            shift_logits = sequence_output[..., :-1, :].contiguous()
            shift_labels = lm_labels_embeddings[..., 1:, :].contiguous()
            loss_fct = CosineSimilarity
            ltr_lm_loss = loss_fct(shift_logits.view(-1), shift_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        if lm_labels is not None:
            lm_labels_embeddings = self.bert.embeddings.word_embeddings(lm_labels)
            # max_fct = CosineSimilarity(dim=1, eps=1e-6)
            # ltr_lm_loss = - torch.mean(max_fct(sequence_output.squeeze(), lm_labels_embeddings.squeeze()))
            loss_fct = MSELoss(reduction='sum')
            ltr_lm_loss = loss_fct(sequence_output.squeeze(), lm_labels_embeddings.squeeze())
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)
"""

def bert_sanity_test():
    model_name = "bert-base-uncased"
    max_length = 80
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModelEmbeddingLoss.from_pretrained(model_name)
    input_ids = tokenizer.encode("I am going with my dog to [MASK] park.",
                                 pad_to_max_length=True,
                                 max_length=max_length,
                                 return_tensors='pt')
    lm_labels = tokenizer.encode("I am going with my dog to the park.",
                                 pad_to_max_length=True,
                                 max_length=max_length,
                                 return_tensors='pt')
    output = model(input_ids=input_ids, lm_labels=lm_labels)
    print("bert loss is %f" % output[0])


def get_nearset_token(preds, nbrs):
    # dists, neighbours = nbrs.kneighbors(preds.reshape(-1, preds.shape[2]))
    # neighbours = neighbours.reshape(preds.shape[0], preds.shape[1])
    index = faiss.IndexFlatL2(nbrs.shape[-1])
    index.add(nbrs)
    D, I = index.search(preds.reshape(-1, preds.shape[2]), 1)
    return I.reshape(preds.shape[0], -1)


def Lbert_Ubert_sanity_test():
    with torch.no_grad():
        max_length = 80
        input_sentence = "the native language of louis - jean - marie daubenton is [MASK]."
        rewrited_sentence = "it seems that the first language of louis - jean - marie daubenton is [MASK]."
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        lower_model = BertModelEmbeddingLoss.from_pretrained('/Users/yuvalkirstain/school/repos/BERTese/BERTese/output/bert-large-uncased-rewrite-cosine-checkpoint-51031')
        upper_model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        vocab = torch.tensor(list(range(upper_model.config.vocab_size)))
        nbrs = upper_model.bert.embeddings.word_embeddings(vocab)
        nbrs = nbrs.detach().numpy()

        input_ids = tokenizer.encode(input_sentence,
                                     pad_to_max_length=True,
                                     max_length=max_length,
                                     return_tensors='pt')
        out_embs = lower_model(input_ids=input_ids)[0]
        out_embs_numpy = out_embs.detach().numpy()

        lower_output_ids = get_nearset_token(out_embs_numpy, nbrs)[0]
        lower_output = tokenizer.decode(lower_output_ids)

        cleaned_lower_output = lower_output[lower_output.index(' '): lower_output.index('[SEP]')].strip()
        print("lower model nn output is: '%s'" % cleaned_lower_output)
        assert cleaned_lower_output == rewrited_sentence

        output = upper_model(inputs_embeds=torch.tensor(out_embs_numpy))[0].detach().numpy()
        predicted = np.argmax(output[0], axis=1)
        print("upper model output with lower output as input is: '%s'" % tokenizer.decode(predicted))

        input_ids = tokenizer.encode(rewrited_sentence,
                                     pad_to_max_length=True,
                                     max_length=max_length,
                                     return_tensors='pt')
        regular_input_ids = torch.tensor(lower_output_ids.reshape(1, -1))
        assert torch.all(input_ids == regular_input_ids)
        regular_output = upper_model(input_ids=regular_input_ids)[0].detach().numpy()
        regular_predcited = np.argmax(regular_output[0], axis=1)
        print("upper model output with nn on lower output as input is '%s'" % tokenizer.decode(regular_predcited, skip_special_tokens=True))
        regular_embs = upper_model.bert.embeddings.word_embeddings(input_ids)
        print("l2 dist of regular embds and lower model output embs is %f" % torch.norm(out_embs - regular_embs))
        output = upper_model(inputs_embeds=regular_embs)[0].detach().numpy()
        predicted = np.argmax(output[0], axis=1)
        print("upper model output with true embedding is : '%s'" % tokenizer.decode(predicted))
        np.random.seed(42)
        for scale in [1, 1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([scale]))
            noise = normal_dist.sample((regular_embs.view(-1).size())).reshape(regular_embs.size())
            noisy_embs = regular_embs + noise
            print("l2 dist is %f" % torch.norm(noise))
            output = upper_model(inputs_embeds=noisy_embs)[0].detach().numpy()
            predicted = np.argmax(output[0], axis=1)
            print("upper model output with a normal noise with mean 0 and scale %f embedding is : '%s'" % (scale, tokenizer.decode(predicted)))


def Lbert_Ubert_base_sanity_test():
    with torch.no_grad():
        max_length = 80
        #input_sentence = "the native language of louis - jean - marie daubenton is [MASK]."
        input_sentence ="iginio ugo tarchetti used to communicate in [MASK]."
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lower_model = BertModelEmbeddingLoss.from_pretrained(
            "/specific/netapp5_2/gamir/adi/BERT_models/bertese_lower_bert_pre_train_models/bert-base-uncased-identity-mse-sum-checkpoint-51031/")
        upper_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        vocab = torch.tensor(list(range(upper_model.config.vocab_size)))
        nbrs = upper_model.bert.embeddings.word_embeddings(vocab)
        nbrs = nbrs.detach().numpy()

        input_ids = tokenizer.encode(input_sentence,
                                     pad_to_max_length=True,
                                     max_length=max_length,
                                     return_tensors='pt')
        out_embs = lower_model(input_ids=input_ids)[0]
        out_embs_numpy = out_embs.detach().numpy()

        lower_output_ids = get_nearset_token(out_embs_numpy, nbrs)[0]
        lower_output = tokenizer.decode(lower_output_ids)

        cleaned_lower_output = lower_output[lower_output.index(' '): lower_output.index('[SEP]')].strip()
        print("lower model NN output is: '%s'" % cleaned_lower_output)
        assert cleaned_lower_output == input_sentence

        output = upper_model(inputs_embeds=torch.tensor(out_embs_numpy))[0].detach().numpy()
        predicted = np.argmax(output[0], axis=1)
        print("upper model output with lower output as input is: '%s'" % tokenizer.decode(predicted))


if __name__ == '__main__':
    # t5_sanity_test()
    # bert2bert_sanity_test()
    # bert_sanity_test()
    #Lbert_Ubert_sanity_test()
    Lbert_Ubert_base_sanity_test()