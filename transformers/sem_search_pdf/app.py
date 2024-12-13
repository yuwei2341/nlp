from flask import Flask, request, render_template, redirect, url_for, g
import os
import logging
import shutil

from werkzeug.utils import secure_filename
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk, concatenate_datasets, Dataset
import torch
import torch.nn.functional as F
from utils.pdf_reader import extract_information
from utils.embedding_generator import compute_embeddings, generate_embeddings_for_dataframe
from utils.file_manager import get_pdf_names, remove_file_and_embedding

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Log all levels (DEBUG and above)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to app.log
        logging.StreamHandler()  # Output logs to console as well
    ]
)
app.logger = logging.getLogger(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_DATA_FOLDER'] = 'extracted_data'
app.config['PDF_DIRECTORY'] = 'data/pdf_files'
app.config['MODEL_CKPT'] = 'Alibaba-NLP/gte-multilingual-base'
app.config['K_NEIGHBORS'] = 5
app.secret_key = 'XXXX'
# Ensure the data and PDF directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_DIRECTORY'], exist_ok=True)

## Initialization of the model and data
# Load the model only once during the app startup.
app.logger.info(f"Loading model {app.config['MODEL_CKPT']} on app startup...")
try:
    tokenizer = AutoTokenizer.from_pretrained(app.config['MODEL_CKPT'])
    model = AutoModel.from_pretrained(app.config['MODEL_CKPT'], trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    app.config.update({
        'MODEL': model,
        'TOKENIZER': tokenizer,
        'DEVICE': device,
    })
    app.logger.info(f"Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {e}")


# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    """Helper function to check allowed file types"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def initialize_globals():
    """
    Initialize global variables before handling each request.

    This function sets up the global variables `pdf_embeddings` and `pdf_names`
    to ensure they are initialized for every request. It loads existing PDF embeddings
    from the extracted data directory if available.
    """
    g.pdf_embeddings = None

    extracted_files = os.listdir(app.config['EXTRACTED_DATA_FOLDER'])
    for file_name in extracted_files:
        file_path = os.path.join(app.config['EXTRACTED_DATA_FOLDER'], file_name)
        loaded_embeddings = load_from_disk(file_path, keep_in_memory=True)
        if g.pdf_embeddings is None:
            g.pdf_embeddings = loaded_embeddings
        else:
            g.pdf_embeddings = concatenate_datasets([g.pdf_embeddings, loaded_embeddings])
    
    g.pdf_names = get_pdf_names(g.pdf_embeddings)


# Home page route
@app.route('/')
def home(): 
    app.logger.info(f"Loaded PDF names: {g.pdf_names}")
    return render_template('home.html', pdf_names=g.pdf_names)

# File upload route
@app.route('/upload', methods=['POST'])
def upload_pdfs():
    """Handles the upload of new PDF files and extracts their text content."""
    uploaded_files = request.files.getlist('pdf_files')

    app.logger.info(f"g.pdf_names: {g.pdf_names}")
    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.filename):
            file_name = secure_filename(uploaded_file.filename)
            if file_name in g.pdf_names:
                app.logger.warning(f"File {file_name} already exists. Prompting user for confirmation...")
                # Store the file temporarily for potential replacement
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                uploaded_file.save(temp_path)
                
                # Render a confirmation page asking for user action
                return render_template('confirm_replace.html', file_name=file_name)
            
            # If the uploaded file is new
            file_path = os.path.join(app.config['PDF_DIRECTORY'], file_name)
            uploaded_file.save(file_path)
            process_and_save_file(file_path, file_name)            
    g.pdf_names = get_pdf_names(g.pdf_embeddings)
    return redirect(url_for('home'))


def process_and_save_file(file_path, file_name):
    """
    Process a newly uploaded PDF file and save its extracted text and
    corresponding embeddings to the extracted data directory.

    Args:
        file_path (str): The file path of the uploaded PDF file.
        file_name (str): The name of the uploaded PDF file.

    Returns:
        None
    """
    extracted_text = extract_information(file_path, file_name)
    new_embeddings = generate_embeddings_for_dataframe(
        extracted_text, app.config["TOKENIZER"], app.config["MODEL"], app.config["DEVICE"],
    )

    if g.pdf_embeddings is None:
        g.pdf_embeddings = new_embeddings
    else:
        g.pdf_embeddings = concatenate_datasets([g.pdf_embeddings, new_embeddings])

    new_embeddings.save_to_disk(os.path.join(app.config["EXTRACTED_DATA_FOLDER"], file_name))
    app.logger.info(f"Uploaded and extracted {file_name}")


@app.route('/replace', methods=['POST'])
def replace_file_confirmation():
    """Handle user's decision to replace or keep the existing file."""
    action = request.form['action']
    file_name = request.form['file_name']

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file_path = os.path.join(app.config['PDF_DIRECTORY'], file_name)
    embedding_path = os.path.join(app.config['EXTRACTED_DATA_FOLDER'], file_name)

    if action == 'yes':
        app.logger.info(f"Replacing file {file_name}.")
        remove_file_and_embedding(file_path, embedding_path)
        os.rename(temp_path, file_path)
        process_and_save_file(file_path, file_name)
        return redirect(url_for('home'))

    elif action == 'no':
        app.logger.info(f"User chose not to replace file {file_name}.")
        os.remove(temp_path)  # Clean up the temporary file
        return redirect(url_for('home'))

    return "Invalid action.", 400


# Search functionality
@app.route('/search', methods=['POST'])
def search():
    """
    Handles the search functionality by computing the embedding of the query text, 
    retrieving the nearest neighbors from the stored PDF embeddings, and rendering the 
    search results.

    Parameters:
        query (str): The search query text.

    Returns:
        A rendered HTML template with the search results.
    """
    query_text = request.form['query']

    query_embedding = compute_embeddings(
        [query_text], app.config["TOKENIZER"], app.config["MODEL"], app.config["DEVICE"],
    ).cpu().detach().numpy()

    selected_pdf_files = request.form.getlist('pdf_files')
    if not selected_pdf_files:
        selected_pdf_files = list(set(g.pdf_embeddings['file_name']))

    search_embeddings = g.pdf_embeddings.filter(lambda x: x['file_name'] in selected_pdf_files)
    search_embeddings.add_faiss_index(column="embeddings")

    scores, samples = search_embeddings.get_nearest_examples(
        "embeddings", query_embedding, k=app.config['K_NEIGHBORS']
    )
    search_results_df = pd.DataFrame.from_dict(samples)
    search_results_df["scores"] = scores
    search_results_df.sort_values("scores", ascending=True, inplace=True)

    search_results_list = search_results_df[
        ['file_name', 'title', 'page_in_pdf', 'text', 'scores']
    ].values.tolist()

    return render_template('home.html', 
                           pdf_names=g.pdf_names, 
                           search_results=search_results_list)


if __name__ == '__main__':
    app.run(debug=True)

