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


def make_predictions(model, tokenizer, paragraphs):
    """
    Return prediction values (using functions defined earlier)
    
    1) Preprocess paragraphs using tokenizer
    2) Predict on dataloader and return prediction values
    
    """
    
    prediction_dataloader = bert_preprocess(paragraphs, tokenizer)        # BERT preprocess
    predictions = eval_cpu(prediction_dataloader, model)                  # Get prediction probabilities
    return predictions
     

def pred_names(predictions):
    """Return a list of predicted names from prediction values."""
    le_classes = ['Emotet', 'Mirai', 'Zeus']    
    predicted_names = []
    
    for i in predictions:
        prediction = le_classes[np.argmax(i)]
        predicted_names.append(prediction)
        
    return predicted_names

