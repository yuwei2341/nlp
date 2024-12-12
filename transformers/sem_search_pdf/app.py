from flask import Flask, request, render_template, redirect, url_for, g
import os
import logging
from werkzeug.utils import secure_filename

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk, concatenate_datasets, Dataset
import torch
import torch.nn.functional as F
from utils.pdf_reader import extract_information
from utils.embedding_generator import compute_embeddings, generate_embeddings_for_dataframe
from utils.file_manager import (
    save_extracted_data_to_csv,
    get_uploaded_files,
    file_exists,
    remove_file,
)

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
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_DATA_FOLDER'] = 'extracted_data'
app.config['PDF_DIRECTORY'] = 'data/pdf_files'
app.secret_key = 'XXXX'

## Initialization of the model and data
# Load the model only once during the app startup.
model_ckpt = 'Alibaba-NLP/gte-multilingual-base'
logger.info(f"Loading model {model_ckpt} on app startup...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    app.config['MODEL'] = model
    app.config['TOKENIZER'] = tokenizer
    app.config['DEVICE'] = device
    logger.info(f"Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")

# Ensure the data and PDF directory exists
os.makedirs(app.config['PDF_DIRECTORY'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

@app.before_request
def before_request():
    """
    Initialize global variables before handling each request.

    This function sets up the global variables `pdf_embeddings` and `pdf_names`
    to ensure they are fresh for every request. It attempts to load existing
    PDF embeddings from a CSV file if available.
    """
    # Initialize pdf_embeddings and pdf_names for each request
    g.pdf_embeddings = None
    g.pdf_names = []
    
    # Check if the embeddings file exists and load it
    extracted_pdf_files = os.listdir(app.config['EXTRACTED_DATA_FOLDER'])
    for file in extracted_pdf_files:
        loaded = load_from_disk(os.path.join(app.config['EXTRACTED_DATA_FOLDER'], file))
        g.pdf_embeddings = loaded if g.pdf_embeddings is None else concatenate_datasets([g.pdf_embeddings, loaded])
        logger.info(f"pdf_embeddings loaded for {file}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page route
@app.route('/')
def home():
    if g.pdf_embeddings is not None:
        g.pdf_names = list(set(g.pdf_embeddings['file_name'])) 
    logger.info(f"pdf_embeddings in home {g.pdf_embeddings}")
    logger.info(f"pdf_names: {g.pdf_names}")
    return render_template('home.html', pdf_names=g.pdf_names)

# File upload route
@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('pdf_files')
    for file in files:
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            if file_name in g.pdf_names:
                # Ask user if they want to replace the file
                return "File already exists. Do you want to replace it? [yes/no]"
            else:
                file_path = os.path.join(app.config['PDF_DIRECTORY'], file_name)
                file.save(file_path)

                # Extract text from the PDF and store it
                df_pdf = extract_information(file_path, file_name)
                # compute_embeddings, 
                new_embeddings = generate_embeddings_for_dataframe(
                    df_pdf, app.config["TOKENIZER"], app.config["MODEL"], app.config["DEVICE"], 
                )
                # logger.info(f"{g.pdf_embeddings.features}, {new_embeddings.features}")
                g.pdf_embeddings = new_embeddings if g.pdf_embeddings is None else concatenate_datasets([g.pdf_embeddings, new_embeddings])
                new_embeddings.save_to_disk(os.path.join(app.config["EXTRACTED_DATA_FOLDER"], file_name))
                logging.info(f"Uploaded and extracted {file_name}")
    return redirect(url_for('home'))

# Search functionality
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    query_embedding = compute_embeddings(
        [query], app.config["TOKENIZER"], app.config["MODEL"], app.config["DEVICE"], 
        ).cpu().detach().numpy()

    selected_pdfs = request.form.getlist('pdf_files')
    if not selected_pdfs:
        selected_pdfs = list(set(g.pdf_embeddings['file_name']))   # Search in all PDFs if none selected
    logger.info(f"selected_pdfs {selected_pdfs}")

    search_pdf_embeddings = g.pdf_embeddings.filter(lambda x: x['file_name'] in selected_pdfs)
    logger.info(f"search_pdf_embeddings {search_pdf_embeddings}")

    search_pdf_embeddings.add_faiss_index(column="embeddings")
    scores, samples = search_pdf_embeddings.get_nearest_examples(
        "embeddings", query_embedding, k=5
    )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=True, inplace=True)

    # search_results = {}
    # for _, row in samples_df.iterrows():
    #     # search_results[pdf_name] = row["text"]
    #     search_results['tmp'] = [row["text"]]
    
    # search_results['aaa'] = 'bbb'
    # return render_template('home.html', 
    #                        pdf_names=g.pdf_names, 
    #                        search_results=search_results)

    logger.info(f"{samples_df.columns}")
    search_results = samples_df[['file_name', 'title', 'paragraph', 'text', 'scores']].values.tolist()
    return render_template('home.html', 
                           pdf_names=g.pdf_names, 
                           search_results=search_results)


if __name__ == '__main__':
    app.run(debug=True)
