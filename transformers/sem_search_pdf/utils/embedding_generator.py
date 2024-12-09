#!/usr/bin/env python3
from datasets import Dataset
# For Type Annotations, better use the base classes below over Auto...
from transformers import PreTrainedTokenizer, PreTrainedModel  
import torch
import pandas as pd

def create_features_from_dataframe(df: pd.DataFrame) -> Dataset:
    """
    Creates a HuggingFace Dataset from a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): The input dataframe containing columns for 'title' and 'text'.
    
    Returns:
        Dataset: A HuggingFace Dataset with a 'data' column combining 'title' and 'text'.
    """
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda x: {'data': f"{x['title']} \n {x['text']}"}
    )
    return dataset


def extract_cls_embedding(model_output) -> torch.Tensor:
    """
    Extracts the [CLS] token embedding from the model output.
    
    Args:
        model_output (ModelOutput): The output from the transformer model.
    
    Returns:
        torch.Tensor: The [CLS] token embedding.
    """
    return model_output.last_hidden_state[:, 0]


def compute_embeddings(
    text_list: list, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, device: torch.device
) -> torch.Tensor:
    """
    Computes the embeddings for a list of texts using the provided tokenizer and model.
    
    Args:
        text_list (list): A list of text strings to compute embeddings for.
        tokenizer (PreTrainedTokenizer): The tokenizer to encode the texts.
        model (PreTrainedModel): The model to generate embeddings.
        device (torch.device): The device (CPU/GPU) on which the model should run.
    
    Returns:
        torch.Tensor: The embeddings for the input texts.
    """
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    model_output = model(**encoded_input)
    return extract_cls_embedding(model_output)


def generate_embeddings_for_dataframe(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizer, device: torch.device, model: PreTrainedModel
) -> Dataset:
    """
    Generates embeddings for a dataframe of text data by combining 'title' and 'text' columns.
    
    Args:
        df (pandas.DataFrame): The dataframe with 'title' and 'text' columns.
        tokenizer (PreTrainedTokenizer): The tokenizer to encode the texts.
        device (torch.device): The device (CPU/GPU) on which the model should run.
        model (PreTrainedModel): The model used to generate embeddings.
    
    Returns:
        Dataset: A HuggingFace Dataset with an 'embeddings' column containing the computed embeddings.
    """
    dataset = create_features_from_dataframe(df)
    embeddings_dataset = dataset.map(
        lambda x: {'embeddings': compute_embeddings(x['data'], tokenizer, model, device).detach().cpu().numpy()[0]}
    )
    return embeddings_dataset
