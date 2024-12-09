from flask import Flask, request, render_template, redirect, url_for
import os
import logging
from werkzeug.utils import secure_filename

import pandas as pd
from transformers import AutoTokenizer, AutoModel
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
app.config['EXTRACTED_DATA'] = None
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

## TODO - load existing dataset of previously extracted data
pdf_data = {}
pdf_directory = 'data/pdf_files'

# # Ensure the data and PDF directory exists
# os.makedirs(pdf_directory, exist_ok=True)
# os.makedirs('logs', exist_ok=True)

# # Load existing PDFs into memory at the start of the app
# def load_pdfs():
#     for pdf_name in os.listdir(pdf_directory):
#         if pdf_name.endswith('.pdf'):
#             with open(os.path.join(pdf_directory, pdf_name), 'rb') as f:
#                 reader = PyPDF2.PdfReader(f)
#                 text = ""
#                 for page in reader.pages:
#                     text += page.extract_text()
#                 pdf_data[pdf_name] = text

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page route
@app.route('/')
def home():
    pdf_names = ['a', 'b']
    # pdf_names = list(pdf_data.keys())
    return render_template('home.html', pdf_names=pdf_names)

# File upload route
@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('pdf_files')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if filename in pdf_data:
                # Ask user if they want to replace the file
                return "File already exists. Do you want to replace it? [yes/no]"
            else:
                file_path = os.path.join(pdf_directory, filename)
                file.save(file_path)

                # Extract text from the PDF and store it
                df_pdf = extract_information(file_path)
                # compute_embeddings, 
                pdf_dataset = generate_embeddings_for_dataframe(
                    df_pdf, app.config["TOKENIZER"], app.config["MODEL"], app.config["DEVICE"], 
                )
                app.config['EXTRACTED_DATA'] = pdf_dataset
                pdf_dataset.save_to_disk(os.path.join(app.config['EXTRACTED_DATA_FOLDER'], filename))

                logging.info(f"Uploaded and extracted {filename}")

                # After uploading, reload the PDFs to include the newly uploaded file
                # load_pdfs()  # This reloads the pdf_data to include new PDF data
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
        selected_pdfs = list(pdf_data.keys())  # Search in all PDFs if none selected

    search_results = {}
    # for pdf_name in selected_pdfs:
        # text = pdf_data.get(pdf_name, "")
        # if query.lower() in text.lower():
        #     paragraphs = [p for p in text.split("\n") if query.lower() in p.lower()]
    pdf_dataset = app.config['EXTRACTED_DATA']
    pdf_dataset.add_faiss_index(column="embeddings")
    scores, samples = pdf_dataset.get_nearest_examples(
        "embeddings", query_embedding, k=5
    )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=True, inplace=True)

    for _, row in samples_df.iterrows():
        # search_results[pdf_name] = row["text"]
        search_results['tmp'] = [row["text"]]
    
    search_results['aaa'] = 'bbb'
    return render_template('home.html', 
                        #    pdf_names=list(pdf_data.keys()), 
                           pdf_names=['tmp', 'aaa'], 
                           search_results=search_results)

if __name__ == '__main__':
    app.run(debug=True)
