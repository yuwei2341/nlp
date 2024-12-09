#!/usr/bin/env python3

import os
import csv

def save_extracted_data_to_csv(data, filename, output_folder):
    csv_filename = os.path.join(output_folder, filename + ".csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Key", "Value"])
        for key, value in data.items():
            writer.writerow([key, value])

def get_uploaded_files(upload_folder):
    return [
        f for f in os.listdir(upload_folder)
        if os.path.isfile(os.path.join(upload_folder, f))
    ]

def file_exists(filepath):
    return os.path.exists(filepath)

def remove_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
