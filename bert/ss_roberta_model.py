import transformers
import torch
import torch.nn as nn
from textblob import TextBlob
print('transformer version:', transformers.__version__)


from transformers import PreTrainedModel, RobertaConfig, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings, RobertaEncoder, RobertaPooler, RobertaClassificationHead #, RobertaEncoder, RobertaPooler,

class RobertaForSequenceClassification_Ss_IDW(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels=None, tokenizer=None, igw_after_chuli=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        #self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None



        self.igw = igw_after_chuli

        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).to(device)





        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        inputids_first_dimension = embedding_output.size()[0]  # batch size
        hidden_dimensions = embedding_output.size()[2]
        Ss = torch.empty(inputids_first_dimension, 1, hidden_dimensions).to(device)  # e.g. [32, 1, 768]
        IDW = torch.empty([inputids_first_dimension, 1], dtype=torch.long).to(device)
        for i, the_id in enumerate(input_ids):
            sent = self.tokenizer.convert_ids_to_tokens(the_id.tolist())
            new_sent = ''
            for word in sent:
                if word != '[PAD]':
                    new_sent = new_sent + word + ' '

            blob = TextBlob(new_sent)
            subjective = blob.sentiment.subjectivity
            Ss[i, 0, :] = subjective
            # Ss size [32, 1, 768]

            sent = [x.lower() for x in sent]
            words = set(sent)
            inter = words.intersection(self.igw)
            if len(inter) > 0:
                IDW[i, 0] = 1
            elif len(inter) == 0:
                IDW[i, 0] = 0

        IDW = IDW.to(torch.device("cuda"))
        Ss = Ss.to(torch.device("cuda"))

        embedding_output = torch.cat([embedding_output, Ss], dim=1) # [32, 128, 768] [32, 1, 768] --> [32, 129, 768]

        # 处理 attention mask
        attention_mask = torch.cat([attention_mask, IDW], dim=1)  # [32, 128] [32, 1] --> [32, 129]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [32, 129] --> [32, 1, 1, 129]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 进 encoder
        encoder_output = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)














        sequence_output = encoder_output[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

