#!/usr/bin/env python3
import shutil
import os
import logging
import csv
from datasets import Dataset


logger = logging.getLogger(__name__)


def remove_file_and_embedding(file_path, embedding_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        shutil.rmtree(embedding_path)
        logger.info(
            f"Removed existing file {file_path} and embeddings {embedding_path}")


def get_pdf_names(dataset: Dataset):
    """Helper function to update the list of PDF names from the embeddings data."""
    if dataset is not None:
        return list(set(dataset["file_name"]))
    return []
