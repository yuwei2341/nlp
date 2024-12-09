#!/usr/bin/env python3
"""
read_pdf.py
----------------
This module contains functions for processing PDF files converted from OneNote pages. 
OneNote pages generally contain a title followed by the date and time the page was created,
with content organized in bullet points. This module provides functionality to extract text 
from a given PDF document. The `pypdf` library is used for PDF reading and extraction.

Functions:
    - extract_information(pdf_path): Extracts textual information from the PDF file located at the 
      given path. Returns the extracted text as a DataFrame or raises an error if the PDF cannot be processed.

Dependencies:
    - pypdf: Required for reading and extracting content from PDF files.
    
Usage:
    - Import this module in other parts of the application where PDF processing is needed.
    - Example usage:
        from utils.read_pdf import extract_information
        extracted_text = extract_information('path/to/file.pdf')

Author: [Your Name]
Date: [Creation Date or Last Update]
"""

import re
import pandas as pd
from pypdf import PdfReader


def find_title(text):
    """
    Extracts the title of the page from the given text. 
    The title is assumed to be the first non-date line above the creation date (which contains the weekday).

    Parameters:
        text (str): The textual content of the PDF page.

    Returns:
        str: The title of the page or None if the title cannot be determined.
    """
    weekday_match = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)', text)
    if weekday_match:
        # Title ends right before the weekday name
        title_end = weekday_match.start() - 1
        # Find the start of the title by searching for the first newline before the weekday
        title_start = text.rfind('\n', 0, title_end) + 1
        title = text[title_start:title_end].strip()
        return title
    return None


def extract_text(file_path):
    """
    Extracts text from a PDF file converted from OneNote pages.
    The PDF content is parsed and split into a DataFrame where each row contains a page title, 
    the corresponding page number, and the paragraphs of text.

    Parameters:
        file_path (str): The file path to the PDF document.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted information with columns: 
                      'page_in_on', 'title', 'page_in_pdf', 'paragraph', 'text'.
    """
    reader = PdfReader(file_path)
    pages = reader.pages

    extracted_data = []  # List to store the extracted data

    current_title = ''
    title_index = 0

    for i, page in enumerate(pages):
        text = page.extract_text()
        title = find_title(text)

        # Update the current title if a new title is found
        if title:
            current_title = title
            title_index += 1

        # Clean the text by removing bullet points and splitting it into paragraphs
        paragraphs = [para for para in re.split(r'\b\w\.\n?', text) if para.strip()]
        
        # Add each paragraph as a new row in the extracted data list
        for j, paragraph in enumerate(paragraphs):
            extracted_data.append((title_index, current_title, i + 1, j + 1, paragraph))

    # Convert the list of extracted data into a pandas DataFrame
    df = pd.DataFrame(extracted_data, columns=['page_in_on', 'title', 'page_in_pdf', 'paragraph', 'text'])
    return df


def extract_information(pdf_path):
    """
    Extracts and processes information from the provided PDF file.

    This function wraps the extraction process, allowing external modules to use it with a single call.
    
    Parameters:
        pdf_path (str): The path to the PDF document.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted text, organized by titles, page numbers, and paragraphs.
    """
    try:
        df = extract_text(pdf_path)
        return df
    except Exception as e:
        raise Exception(f"An error occurred while extracting information from the PDF: {str(e)}")
