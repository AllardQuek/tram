import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


def load_bert(filepath):
    """Load BERT tokenizer and pre-trained BERT model"""
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print('Loading BERT model...')
    model = BertForSequenceClassification.from_pretrained(filepath)

    return model, tokenizer


def preprocess_sents(sentences_list):
    """Clean up sentences predicted by TRAM"""
    prepocessed_sents = []
    for s in sentences_list:
        # Replace any new lines separating parts of the sentence
        s = s.replace('\n', ' ')

        # Replace any double spaces which might result from previous step with a single space
        s = s.replace('  ', ' ')

        # Do a length check to skip empty strings and random punctuation
        if len(s) < 3:
            continue
        prepocessed_sents.append(s)
    return prepocessed_sents


def bert_preprocess(paragraphs, tokenizer):
    """Conduct preprocessing requried for BERT and return the dataloader (MAX_LEN = 256).
    
    Parameters --
        :paragraphs: array of praragraphs (or sentences)
        :tokenizer: loaded BERT tokenizer

    Returns --
        :prediction_dataloader: Pytorch dataloader to store prediction data
    """
    MAX_LEN = 256

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids, attention_masks = [], []

    for para in paragraphs:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            para,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 256,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Set the batch size, same as that used during training
    batch_size = 5

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)    
    prediction_dataloader = DataLoader(
        prediction_data,
        sampler = SequentialSampler(prediction_data),
        batch_size = batch_size
    )

    return prediction_dataloader


def eval_cpu(prediction_dataloader, model):
    """
    Predict using CPU and return an array of prediction indexes.

    Parameters --
        :prediction_dataloader: dataloader to be unloaded, from pre-processing step
        :model: loaded BERT model
    
    Returns -- 
        :flat_predictions: array of prediction indexes (obtained from argmax of prediction values)
    """
    # Put model in evaluation mode
    model.eval()       

    predictions = []

    # Predict 
    for batch in prediction_dataloader:
        batch = tuple(t for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        b_input_ids = b_input_ids.type(torch.LongTensor)

        # Tell model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        # Since we only have 1 batch, we can simply return logits, else we would append each batch's logits to a list and return that list
        logits = outputs[0]     

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    # Combine the results across all batches (stack batches one on top of another)
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    return flat_predictions


def get_pred_ids(predictions):
    """Return a list of predicted malware ids from prediction values."""
    le_classes = ['Emotet', 'Mirai', 'Zeus']    
    malwares_dict = {'Emotet': 1, 'Mirai': 2, 'Zeus': 3}
    predicted_ids = []
    
    for idx in predictions:
        pred_name = le_classes[idx]
        pred_id = malwares_dict[pred_name]
        predicted_ids.append(pred_id)
        
    return predicted_ids


def make_predictions(model, tokenizer, paragraphs):
    """
    Return predicted malware ids (using functions defined earlier)
    
    1) Preprocess paragraphs using tokenizer
    2) Predict on dataloader and return prediction values
    3) Derive malware ids from predicted values
    
    """
    
    prediction_dataloader = bert_preprocess(paragraphs, tokenizer)        
    predictions = eval_cpu(prediction_dataloader, model)                  
    malware_ids = get_pred_ids(predictions)                              
    return malware_ids
    

def write_data(tech_id, tech_name, sentence, source, date_crawled):
    """Write output to file (alternative to inserting to database)"""
    with open('PDF_data.txt', 'a') as f:
        # text = match["tid"] + '\n' + match["name"] + '\n' + sent + '\n' + source + '\n' + date_crawled + '\n\n'
        text = tech_id + '\n' + tech_name + '\n' + sentence + '\n' + source + '\n' + date_crawled + '\n\n'
        f.write(text)