#!/usr/bin/env python3
# Medical Report OCR Parser with Ollama DeepSeek + Python Regression Validation System (Final Stable Version)
# Author: Konda Charana

import os
import cv2
import numpy as np
import json
import requests
import glob
import traceback
import difflib
import pandas as pd
import shutil
import re
import time
from datetime import datetime
from tqdm import tqdm
import pytesseract

# ====== CONFIGURATION VARIABLES ======
INPUT_FOLDER = "./input_images"
OUTPUT_FOLDER = "./output"
MAX_IMAGES = 0
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
DEBUG_MODE = True
OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 1024,
    "num_ctx": 4096
}
# =====================================

REFERENCE_FOLDER = "./reference_jsons"
VALIDATION_REPORT_FOLDER = "./output/validation_reports"

class MedicalReportOCR:
    def __init__(self, ollama_url=OLLAMA_BASE_URL, model_name=OLLAMA_MODEL):
        self.ollama_url = ollama_url
        self.model_name = model_name

        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {ollama_url}")
                models = [m['name'] for m in response.json().get('models', [])]
                if model_name in models:
                    print(f"✅ Model {model_name} is available")
                else:
                    print(f"⚠️ Model {model_name} not found. Run: ollama pull {model_name}")
            else:
                print(f"❌ Failed to connect to Ollama")
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharpened)
            return enhanced
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            raise

    def extract_text_tesseract(self, image_path):
        try:
            processed_img = self.preprocess_image(image_path)
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            extracted_texts = [
                {'text': t.strip(), 'confidence': int(c)}
                for t, c in zip(data['text'], data['conf']) if t.strip() and int(c) > 30
            ]
            full_text = ' '.join([i['text'] for i in extracted_texts])
            return full_text, extracted_texts
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return "", []

    # 🧠 UPDATED FUNCTION
    def generate_json_with_ollama(self, extracted_text, image_filename):
        # Limit input size for model stability
        if len(extracted_text) > 8000:
            extracted_text = extracted_text[:8000] + "\n[TRUNCATED TEXT FOR STABILITY]"

        prompt = f"Convert the following extracted medical report text to structured JSON:\n{extracted_text}"
        for attempt in range(2):  # retry twice
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": OLLAMA_OPTIONS
                    },
                    timeout=300
                )
                break
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Ollama connection failed (attempt {attempt+1}): {e}")
                if attempt == 1:
                    return {'success': False, 'error': str(e)}
                print("⏳ Retrying in 5 seconds...")
                time.sleep(5)

        if response.status_code != 200:
            return {'success': False, 'error': f"Ollama HTTP {response.status_code}"}

        #print(f"Raw Ollama response: {response.text[:500]}")

        try:
            result = response.json()
            raw_text = result.get("response", "").strip()

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON structure found in Ollama response")

            json_text = json_match.group(0)

            # Clean invalid characters
            json_text = re.sub(r'[\x00-\x1f\x7f]', '', json_text)

            # 🩹 Repair incomplete braces/brackets
            open_braces = json_text.count('{')
            close_braces = json_text.count('}')
            open_brackets = json_text.count('[')
            close_brackets = json_text.count(']')
            if open_braces > close_braces:
                json_text += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                json_text += ']' * (open_brackets - close_brackets)

            parsed_json = json.loads(json_text)

        except json.JSONDecodeError:
            print("⚠️ JSON parse failed — retrying with truncated text...")
            trimmed_text = json_text.split('```')[0]
            json_match = re.search(r"\{.*\}", trimmed_text, re.DOTALL)
            if json_match:
                clean_text = re.sub(r'[\x00-\x1f\x7f]', '', json_match.group(0))
                open_b = clean_text.count('{')
                close_b = clean_text.count('}')
                if open_b > close_b:
                    clean_text += '}' * (open_b - close_b)
                parsed_json = json.loads(clean_text)
            else:
                raise ValueError("❌ Unable to recover valid JSON structure after truncation.")

        parsed_json['_metadata'] = {
            'source_image': image_filename,
            'timestamp': datetime.now().isoformat()
        }

        return {'success': True, 'json_data': parsed_json}

    def process_image(self, image_path):
        image_filename = os.path.basename(image_path)
        print(f"📄 Processing: {image_filename}")
        extracted_text, details = self.extract_text_tesseract(image_path)
        if not extracted_text.strip():
            return {'success': False, 'error': 'No text extracted'}
        ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)
        if ollama_result['success']:
            return {'success': True, 'structured_json': ollama_result['json_data']}
        else:
            return {'success': False, 'error': ollama_result['error']}

def get_image_files(folder):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def main():
    if not os.path.exists(INPUT_FOLDER):
        print("❌ Input folder not found")
        return

    json_dir = os.path.join(OUTPUT_FOLDER, "json")
    os.makedirs(json_dir, exist_ok=True)

    image_files = get_image_files(INPUT_FOLDER)
    if not image_files:
        print("❌ No image files found")
        return

    ocr = MedicalReportOCR()
    for image_path in tqdm(image_files, desc="Processing Images"):
        result = ocr.process_image(image_path)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(json_dir, f"{base}_extracted.json")
        if result['success']:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result['structured_json'], f, indent=2, ensure_ascii=False)
            print(f"✅ Saved JSON: {out_path}")
        else:
            print(f"❌ Failed: {result['error']}")

# ====== PYTHON REGRESSION VALIDATION SYSTEM ======
def compare_json_structures(new_json, ref_json, path=""):
    mismatches = []
    if isinstance(new_json, dict) and isinstance(ref_json, dict):
        all_keys = set(new_json.keys()) | set(ref_json.keys())
        for key in all_keys:
            n, r = new_json.get(key), ref_json.get(key)
            new_path = f"{path}.{key}" if path else key
            if isinstance(n, (dict, list)) or isinstance(r, (dict, list)):
                mismatches.extend(compare_json_structures(n, r, new_path))
            elif n != r:
                mismatches.append({"path": new_path, "reference": r, "new": n})
    elif isinstance(new_json, list) and isinstance(ref_json, list):
        for i, (n, r) in enumerate(zip(new_json, ref_json)):
            mismatches.extend(compare_json_structures(n, r, f"{path}[{i}]"))
    else:
        if new_json != ref_json:
            mismatches.append({"path": path, "reference": ref_json, "new": new_json})
    return mismatches

def compute_text_similarity(text1, text2):
    return round(difflib.SequenceMatcher(None, str(text1), str(text2)).ratio() * 100, 2)

def run_regression_validation(new_path, ref_path):
    with open(new_path, 'r', encoding='utf-8') as f1, open(ref_path, 'r', encoding='utf-8') as f2:
        new_json, ref_json = json.load(f1), json.load(f2)
    mismatches = compare_json_structures(new_json, ref_json)
    similarity = compute_text_similarity(json.dumps(new_json), json.dumps(ref_json))
    report = {
        "new_file": os.path.basename(new_path),
        "reference_file": os.path.basename(ref_path),
        "similarity_percent": similarity,
        "total_mismatches": len(mismatches),
        "timestamp": datetime.now().isoformat(),
        "mismatches": mismatches
    }
    os.makedirs(VALIDATION_REPORT_FOLDER, exist_ok=True)
    report_path = os.path.join(VALIDATION_REPORT_FOLDER, f"validation_{os.path.basename(new_path)}")
    with open(report_path, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)
    print(f"📊 Validation Report: {report_path}")
    print(f"   🔍 Similarity: {similarity}% | Mismatches: {len(mismatches)}")
    return report

def batch_validate_results():
    os.makedirs(VALIDATION_REPORT_FOLDER, exist_ok=True)
    json_dir = os.path.join(OUTPUT_FOLDER, "json")
    if not os.path.exists(REFERENCE_FOLDER):
        print("📂 Reference folder missing — creating baseline...")
        os.makedirs(REFERENCE_FOLDER, exist_ok=True)
        for f in glob.glob(os.path.join(json_dir, "*.json")):
            shutil.copy(f, REFERENCE_FOLDER)
        print("✅ Baseline reference set created. Future runs will compare outputs.")
        return

    new_files = glob.glob(os.path.join(json_dir, "*.json"))
    refs = {os.path.basename(r): r for r in glob.glob(os.path.join(REFERENCE_FOLDER, "*.json"))}
    results = []
    for nf in new_files:
        base = os.path.basename(nf)
        if base in refs:
            report = run_regression_validation(nf, refs[base])
            results.append(report)

    if results:
        df = pd.DataFrame([{
            "File": r["new_file"],
            "Similarity (%)": r["similarity_percent"],
            "Mismatches": r["total_mismatches"],
            "Timestamp": r["timestamp"]
        } for r in results])
        summary_csv = os.path.join(VALIDATION_REPORT_FOLDER, "validation_summary.csv")
        df.to_csv(summary_csv, index=False)
        print(f"📑 Summary saved: {summary_csv}")
    else:
        print("⚠️ No reference matches found.")

# ====== VISUAL DIFFERENCE REPORT ======
def generate_visual_diff_report():
    report_files = glob.glob(os.path.join(VALIDATION_REPORT_FOLDER, "validation_*.json"))
    if not report_files:
        print("⚠️ No validation reports found to visualize.")
        return

    all_mismatches = []
    for rpt in report_files:
        with open(rpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        mismatches = data.get("mismatches", [])
        for m in mismatches:
            all_mismatches.append({
                "File": os.path.basename(rpt),
                "JSON Path": m.get("path", ""),
                "Reference Value": str(m.get("reference", "")),
                "New Value": str(m.get("new", "")),
                "Similarity (%)": data.get("similarity_percent", "")
            })

    if not all_mismatches:
        print("✅ No mismatches found across all files.")
        return

    df = pd.DataFrame(all_mismatches)

    # ===== HTML Styling for Color Coding =====
    html_style = """
    <html>
    <head>
        <title>Validation Differences Report</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f6f8; margin: 30px; }
            h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            th { background: #222; color: #fff; padding: 10px; text-align: left; }
            td { padding: 8px; border-bottom: 1px solid #ddd; }
            tr:hover { background-color: #f1f1f1; }
            .match { background-color: #eafbea; }
            .mismatch { background-color: #ffeaea; }
        </style>
    </head>
    <body>
        <h2>📊 Validation Differences Report</h2>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <table>
            <tr>
                <th>File</th>
                <th>JSON Path</th>
                <th>Reference Value</th>
                <th>New Value</th>
                <th>Similarity (%)</th>
            </tr>
    """

    # Build HTML rows
    for _, row in df.iterrows():
        css_class = "mismatch" if row["Reference Value"] != row["New Value"] else "match"
        html_style += f"""
            <tr class="{css_class}">
                <td>{row['File']}</td>
                <td>{row['JSON Path']}</td>
                <td>{row['Reference Value']}</td>
                <td>{row['New Value']}</td>
                <td>{row['Similarity (%)']}</td>
            </tr>
        """

    html_style += """
        </table>
    </body>
    </html>
    """

    html_path = os.path.join(VALIDATION_REPORT_FOLDER, "validation_differences_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_style)

    print(f"📊 Color-coded visual report saved: {html_path}")


if __name__ == "__main__":
    print("🚀 Starting Medical Report OCR + Regression Validation System")
    main()
    print("\n🧩 Running Regression Validation...")
    batch_validate_results()
    print("\n📈 Generating visual HTML report...")
    generate_visual_diff_report()
