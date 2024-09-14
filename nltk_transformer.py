"""
This project is attempting to find the smallest LLM (whether it is an encoder, decoder, or encoder-decoder model) 
that can perform existing, hand-crafted NLP tasks, e.g. tokenizing (using nltk's tokenizer), regex, etc.

"""

from typing import Optional
import nltk
# required to download at least once
# nltk.download('punkt')
# nltk.download('punkt_tab')

import re

import numpy as np
from sklearn.metrics import classification_report
import torch

import tqdm
import transformers
from transformers import (
    BertForTokenClassification,
    BertConfig, 
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset
import transformers.modeling_outputs

TRAIN_SAMPLES = 1200000
EVAL_SAMPLES = 4000
SEED = 42

LABELS = ['middle-of-token', 'end-of-token']

MAX_SEQ_LEN = 256 # this includes the EOS token

def nltk_tokenize(text):
    """
    Tokenizes the text using nltk's word tokenizer (with some modification for quotes), 
    except for whitespace, where each single space, tab, etc. is treated as it's own token
    """

    # split on any whitespace character, also keeping the whitespace characters as tokens 
    tokens = re.split(r'(\s+)', text)

    # at this point though, some whitespace tokens contain multiple characters e.g. '  ', but I only want 1 char/whitespace token like ' ', ' '
    new_tokens = []
    for token in tokens:
        if token.isspace():
            for char in token:
                new_tokens.append(char)
        else:
            new_tokens.append(token)
    tokens = new_tokens

    # now tokenize each non-whitespace token using nltk's word tokenizer
    final_tokens = []
    for token in tokens:
        if token.isspace():
            final_tokens.append(token)
        else:
            final_tokens.extend(nltk.word_tokenize(token))

    # nltk also has an annoying 'feature' where it converts double quotes to either `` or '' in a destructive manner, but I can't have that
    # so I need to go through all of the tokens, check if it *should* be double quotes (and isn't) and update the tokens if that is the case
    for i, token in enumerate(final_tokens):
        if token in ['``', "''"] and (i == 0 or final_tokens[i-1] != '"') and (i == len(final_tokens) - 1 or final_tokens[i+1] != '"'):
            final_tokens[i] = '"'

    return final_tokens
    

# def label_text(text: str):
#     """Labels the text w/ nltk tokens. Each character is labelled either 0 (not the end of an nltk token) or 1 (end of an nltk token)
#     whitespace and other 1-character tokens are labelled as 1."""
#     tokens = nltk_tokenize(text)
#     labels = [0] * len(text)
    
#     index = 0
#     for token in tokens:
#         index += len(token) - 1
#         labels[index] = 1
#         index += 1
    
#     return labels
    
def label_text(text: str) -> list[str]:
    """This is like function above, but it uses a different thing under the hood b/c the default tokenizer has a lot of problems for this application"""
    spans = nltk.tokenize.NLTKWordTokenizer().span_tokenize(text)

    labels = np.ones((len(text),), dtype=np.long) * -1 # unlabelled stuff will be -1 for now, will set to 1 later

    for span in spans:
        labels[span[0]:span[1] - 1] = 0 # set everything within the span to 0 
        labels[span[1] - 1] = 1 # then set the end of the span properly (the upper range of the span is exclusive, hence -1)

    labels[labels == -1] = 1 # anything that isn't in a span is probably whitespace, so those can all be labelled 

    return labels

def byte_tokenize_fn(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, max_length=MAX_SEQ_LEN, padding='do_not_pad')

def chunk_text_to_seq_len(examples, seq_len):
    results = []

    for text, textbook in zip(examples['text'], examples['textbook']):
        # since we label a word/token by it's last character, we have to be careful how we tokenize the text so as not to loose 
        # a label if the word gets truncated halfway through, which is why we truncate all texts ourselves to MAX_SEQ_LEN-1 (the -1 accounts for the EOS token)
        for i in range(0, len(text), seq_len - 1): 
            results.append(text[i:i + seq_len])
        
        for i in range(0, len(textbook), seq_len - 1):
            results.append(textbook[i:i + seq_len])

    examples['text'] = results

    return examples

def create_bert_model(vocab_size=256, hidden_size=384, num_hidden_layers=6, num_attention_heads=12, intermediate_size=1536, class_weights=None):    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        pad_token_id=26, # in ascii this is the SUB character, which I will use as padding
        num_labels=2,  # the only two choices are not end of nltk token and end of nltk token
    )

    model = BertForTokenClassification(config)

    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    def bert_forward_with_class_weights(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    return model

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inference(model, text):
    tokenized = byte_tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**tokenized)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    characters = byte_tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])
    
    result = []
    token = ''
    for character, pred in zip(characters, predictions[0]):
        token += character
        if pred == 1:  # End of token
            result.append(token)
            token = ''
            
    
    return result

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    labels = [
        [l for l in label if l != -100]
        for label in labels
    ]

    # turns out classification report doesn't support sequences of sequences, so make predictions and labels flat
    labels = [item for sublist in labels for item in sublist]
    predictions = [item for sublist in predictions for item in sublist]


    # calculate precision, recall, f1, and accuracy
    results = classification_report(labels, predictions, output_dict=True)

    # for HF, we need to return a single dict for all of the elements
    return {
        'precision': results['weighted avg']['precision'],
        'recall': results['weighted avg']['recall'],
        'f1': results['weighted avg']['f1-score'],
        'accuracy': results['accuracy']
    }

if __name__ == '__main__':  

    byte_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small', clean_up_tokenization_spaces=False)
    byte_tokenizer.pad_token_id = 26
    byte_tokenizer.eos_token_id = 3  # in ascii this is the ETX character, which I will use as the end-of-sequence token


    ### DATASET PREPROCESSING ###

    # this is a small enough dataset to load easily for prototyping but also large enough that we could train for quite some time
    tiny_textbooks = load_dataset('nampdn-ai/tiny-textbooks')

    tiny_textbooks = tiny_textbooks.map(
        chunk_text_to_seq_len,
        fn_kwargs={'seq_len': MAX_SEQ_LEN},    
        batched=True,
        num_proc=8,
        remove_columns=['source', 's', 'len', 'idx', 'textbook'],
        desc="chunking text...",
    )

    print(tiny_textbooks)

    tiny_textbooks = tiny_textbooks.shuffle(seed=42)
    tiny_textbooks['train'] = tiny_textbooks['train'].select(range(TRAIN_SAMPLES))
    tiny_textbooks['test'] = tiny_textbooks['test'].select(range(EVAL_SAMPLES))

    tiny_textbooks = tiny_textbooks.map(
        lambda examples: {'labels': [label_text(text) for text in examples['text']]}, 
        batched=True, 
        num_proc=8,
        desc='labelling text...'
    ).map(
        byte_tokenize_fn,
        fn_kwargs={'tokenizer': byte_tokenizer},
        batched=True, # the collator will take care of padding
        num_proc=8,
        desc='tokenizing text...'
    )

    # also calculate the class weights so that we can fight the class imbalance here
    train_labels = [item for sublist in tqdm.tqdm(tiny_textbooks['train']['labels'], desc='calculating class imbalance...') for item in sublist]
    class_weights = torch.tensor(
        [1 / (train_labels.count(0) / len(train_labels)), 1 / (train_labels.count(1) / len(train_labels))]
    ).to('cuda')

    print(class_weights)

    collator = DataCollatorForTokenClassification(tokenizer=byte_tokenizer, padding='longest', max_length=MAX_SEQ_LEN)


    ### MODEL CREATION ###

    model = create_bert_model(num_hidden_layers=4, hidden_size=96, num_attention_heads=12, intermediate_size=192)
    #model = create_bert_model(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, intermediate_size=32, class_weights=class_weights)

    print(model.num_labels)
    
    print(f"model has {num_parameters(model)} parameters")

    ### TRAINING ###

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=64,
        learning_rate=5e-4,
        num_train_epochs=3,
        seed=SEED,
        eval_strategy='steps',
        eval_steps=.2,
        save_strategy='epoch',
        logging_steps=100,
        bf16=True,
        lr_scheduler_type='cosine',
        dataloader_num_workers=8,
    )

    from AdEMAMix import AdEMAMix

    # optimizer = AdEMAMix(model.parameters(), lr=training_args.learning_rate)
    # scheduler = transformers.get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=100,
    #     num_training_steps=len(tiny_textbooks['train']) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * torch.cuda.device_count())
    # )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tiny_textbooks['train'],
        eval_dataset=tiny_textbooks['test'],
        tokenizer=byte_tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        #optimizers=(optimizer, scheduler),
    )

    trainer.train()


    test_sentence = "This is a test sentence.  It's got some punctuation and whitespace."
    
    print(nltk_tokenize(test_sentence))
    print(inference(trainer.model, test_sentence))