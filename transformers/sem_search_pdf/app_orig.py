import logging, os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
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
# from utils.processing import process_extracted_info, summarize_extracted_info

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
app.secret_key = 'XXXX'

# Load the model only once during the app startup.
model_ckpt = 'Alibaba-NLP/gte-multilingual-base'
logger.info(f"Loading model {model_ckpt} on app startup...")
try:
    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # model = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # app.config['MODEL'] = model
    # app.config['TOKENIZER'] = tokenizer
    # app.config['DEVICE'] = device
    logger.info(f"Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_DATA_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-query', methods=['POST'])
def submit_query():
    query = request.form['query']
    if query:
        # Process the query (you could integrate some NLP model or logic here)
        logger.info(f"Received query: {query}")
        return render_template('query_result.html', query=query, result="Query results here.")  # Replace with actual logic

    flash("Query is required!")
    return redirect(url_for('index'))

@app.route('/upload-pdf', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        logging.warning("No file part in request.")
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        logging.warning("No file selected by the user.")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if file_exists(filepath):
            flash(f'The file "{filename}" already exists. Do you want to replace it?')
            logging.info(f"File {filename} already exists, asking user to replace it.")
            return render_template('index.html', replace_file=filename)

        # Save the new file
        file.save(filepath)
        logging.info(f"File {filename} uploaded successfully.")
        
        # TODO - the embedding computation here
        # # Extract information from the PDF
        # extracted_info = extract_information(filepath)
        # logging.info(f"Extracted data from {filename}: {extracted_info}")

        # # Process extracted info (example: uppercase transformation)
        # processed_info = process_extracted_info(extracted_info)
        # logging.debug(f"Processed information for {filename}: {processed_info}")

        # # Save the extracted data to a CSV file
        # save_extracted_data_to_csv(extracted_info, filename, app.config['EXTRACTED_DATA_FOLDER'])
        # logging.info(f"Extracted data for {filename} saved to CSV.")
        
        flash(f'File "{filename}" uploaded and processed successfully!')
        return render_template('pdf_result.html', extracted_data='xxxx')
    else:
        flash('Invalid file type. Only PDFs are allowed.')
        logging.error("Invalid file type uploaded. Only PDFs are allowed.")
        return redirect(request.url)


@app.route('/pdf-result')
def pdf_result():
    # This is a placeholder route to view the result
    return render_template('pdf_result.html', extracted_data=None)


@app.route('/replace/<filename>', methods=['POST'])
def replace_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    remove_file(filepath)  # Remove the old file
    logging.info(f"Old file {filename} replaced.")

    file = request.files['file']
    file.save(filepath)
    logging.info(f"New file {filename} uploaded to replace the old one.")
    
    # Extract new information from the PDF
    extracted_info = extract_information(filepath)
    logging.info(f"Extracted new data from {filename}: {extracted_info}")

    # # Process the new extracted info
    # processed_info = process_extracted_info(extracted_info)
    # logging.debug(f"Processed new information for {filename}: {processed_info}")

    # Save the extracted data to a CSV file
    save_extracted_data_to_csv(extracted_info, filename, app.config['EXTRACTED_DATA_FOLDER'])
    logging.info(f"New extracted data for {filename} saved to CSV.")
    
    flash(f'File "{filename}" replaced and processed successfully!')
    return redirect(url_for('index'))

@app.route('/view/<filename>', methods=['GET'])
def view_file(filename):
    csv_filepath = os.path.join(app.config['EXTRACTED_DATA_FOLDER'], filename + ".csv")
    try:
        with open(csv_filepath, 'r') as f:
            extracted_info = f.read()
        logging.info(f"Displayed extracted information for {filename}.")
    except Exception as e:
        logging.error(f"Error reading CSV for {filename}: {e}")
        extracted_info = "Error loading file."

    return render_template('index.html', extracted_info=extracted_info)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query']
    selected_files = request.form.getlist('selected_files')  # Get list of selected files
    all_files = request.form.get('select_all_files')

    extracted_info = {}
    logging.info(f"Query received: {query_text}")

    if selected_files:
        # Show extracted info for selected files
        for filename in selected_files:
            csv_filepath = os.path.join(app.config['EXTRACTED_DATA_FOLDER'], filename + ".csv")
            try:
                with open(csv_filepath, 'r') as f:
                    info = f.read()
                logging.info(f"Extracted data displayed for {filename}.")
            except Exception as e:
                logging.error(f"Error reading CSV for {filename}: {e}")
                info = "Error loading file."

            # Process the extracted info
            processed_info = ''  # process_extracted_info(info)
            summary = ''  # summarize_extracted_info(info)

            extracted_info[filename] = {
                "processed": processed_info,
                "summary": summary
            }

    elif all_files == 'true':
        # Show extracted info for all files
        uploaded_files = get_uploaded_files(app.config['UPLOAD_FOLDER'])
        for filename in uploaded_files:
            csv_filepath = os.path.join(app.config['EXTRACTED_DATA_FOLDER'], filename + ".csv")
            try:
                with open(csv_filepath, 'r') as f:
                    info = f.read()
                logging.info(f"Extracted data displayed for all files.")
            except Exception as e:
                logging.error(f"Error reading CSV for {filename}: {e}")
                info = "Error loading file."

            # Process the extracted info
            processed_info = process_extracted_info(info)
            summary = summarize_extracted_info(info)

            extracted_info[filename] = {
                "processed": processed_info,
                "summary": summary
            }

    logging.info(f"Query results returned: {extracted_info}")
    return render_template('index.html', extracted_info=extracted_info, query=query_text)

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True)
