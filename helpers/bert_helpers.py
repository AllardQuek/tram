import torch
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
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
    """Conduct preprocessing requried for BERT and return the dataloader."""
    MAX_LEN = 256

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    for para in paragraphs:
        encoded_para = tokenizer.encode(
                            para,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    )
        
        input_ids.append(encoded_para)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                            dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)

    # Set the batch size. Here we set it equal to number of samples so we have only 1 batch of predictions
    batch_size = len(paragraphs) 

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)    # , prediction_labels
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    print('Continue predicting labels for {:,} prediction sentences...'.format(len(prediction_inputs)))

    return prediction_dataloader


def eval_cpu(prediction_dataloader, model):
    """Unload the dataloader, predict, then return logits which is an array of predictions."""
    # Put model in evaluation mode
    model.eval()       

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
            
    return logits


def get_pred_ids(predictions):
    """Return a list of predicted malware ids from prediction values."""
    le_classes = ['Emotet', 'Mirai', 'Zeus']    
    malwares_dict = {'Emotet': 1, 'Mirai': 2, 'Zeus': 3}
    predicted_ids = []
    
    for i in predictions:
        pred_name = le_classes[np.argmax(i)]
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