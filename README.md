<p align="center">
  <img src="A_banner_image_for_a_Medical_Report_OCR_&_Regressi.png" width="900">
</p>

Medical Report OCR & Regression Validation Framework (FastAPI + Ollama + Tesseract)**

# 🧠 Medical Report OCR & Regression Validation Framework

### **Tesseract OCR + Ollama LLM + FastAPI + Automated Regression System**

This project is a **production-grade AI pipeline** that extracts structured medical information from medical report images using **OCR + LLM reasoning**, and validates outputs using a **Python Regression Validation Framework**.

It exposes the entire system as a **FastAPI microservice**, ready for deployment on **local machines, Docker, or Google Cloud Run**.

---

## 🚀 **Key Features**

### 📌 **1. Advanced OCR Pipeline (Tesseract + OpenCV)**

* CLAHE contrast enhancement
* Denoising + sharpening
* Confidence-filtered text extraction
* Handles low-quality hospital reports

### 🤖 **2. Structured JSON Extraction via Ollama LLM**

* Converts raw OCR text into:

  * Hospital info
  * Patient details
  * Doctor details
  * Test results
  * Additional notes
* Works fully offline (local LLM)

### 🧪 **3. Python Regression Validation System**

Ensures every new OCR+LLM JSON output matches expected baseline:

* JSON structure comparison
* Text similarity scoring
* Mismatch reporting
* Automated baseline creation
* Generates CSV & HTML visual reports

### 🌐 **4. FastAPI REST Service**

Fully production-ready API:

* `/process` → Upload image → Receive structured JSON + validation summary
* `/` → Health check endpoint
* Swagger UI (auto-generated)

### 🐳 **5. Docker Support**

* Lightweight production Dockerfile
* Deployable to Cloud Run, EC2, Kubernetes

---

# 🏗️ **Project Architecture**

```
medical_ocr_api/
│
├── app.py                    # FastAPI service
│
├── ocr_engine/
│     ├── processor.py        # OCR + LLM processing pipeline
│     └── validator.py        # Regression validation engine
│
├── input_images/             # Uploaded images
├── reference_jsons/          # Baseline JSONs (first-run auto-generated)
├── output/
│     ├── json/               # New outputs
│     ├── validation_reports/ # Regression reports
│
├── requirements.txt
└── Dockerfile
```

---

# ⚙️ **Installation & Setup**

### 1️⃣ Clone the repository

git clone https://github.com/KondaCharana/Medical_Report_OCR_and_Regression_Validation_Framework.git
cd Medical_Report_OCR_and_Regression_Validation_Framework


### 2️⃣ Create a virtual environment

python -m venv env
source env/bin/activate       # Linux/Mac
env\Scripts\activate          # Windows


### 3️⃣ Install dependencies

pip install -r requirements.txt


### 4️⃣ Install Tesseract OCR (required)

**Windows:**
Download from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

**Ubuntu:**

```bash
sudo apt install tesseract-ocr
```

### 5️⃣ Ensure Ollama is running

ollama serve
ollama pull llama3.2:3b


---

# 🚀 **Run FastAPI Service**


uvicorn app:app --reload


Open Swagger UI:

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Here you can upload a `.jpg/.png` medical report image and get:

* Extracted structured JSON
* Regression validation summary

---

# 🧪 **Sample Request (cURL)**


curl -X POST "http://127.0.0.1:8000/process" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_report.jpg"


# 🧬 **How the Regression Validation Engine Works**

### 🔍 1. On first run:

* No reference JSON exists
* The framework **auto-creates baseline JSON**

### 🔍 2. On next runs:

* Compares new output vs reference JSON
* Computes similarity score
* Detects mismatched fields
* Saves:

  * Detailed JSON difference report
  * CSV summary
  * HTML visual report

### Example Output:

```json
{
  "file": "report1.jpg",
  "similarity_percent": 86.73,
  "mismatches": 4,
  "validated_at": "2025-11-12T14:05:32"
}
```

---

# 🐳 **Docker Deployment**

### Build

```bash
docker build -t medical-ocr-api .
```

### Run

```bash
docker run -p 8080:8080 medical-ocr-api
```

### Deploy to Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/<project-id>/ocr-api
gcloud run deploy ocr-api --image gcr.io/<project-id>/ocr-api --platform managed --allow-unauthenticated
```

---

# 🧠 **Interview-Ready Explanation**

> *“I designed a medical report OCR pipeline that combines OpenCV preprocessing, Tesseract OCR, and a local Ollama LLM to generate highly structured medical JSON data.
> I implemented a Python regression validation system to ensure output stability and detect changes in OCR performance.
> The entire system runs as a FastAPI microservice, containerized via Docker and deployable to Google Cloud Run.”*

This shows:

* MLOps thinking
* API design
* Validation engineering
* CV + LLM hybrid system
* Deployment capability

---

# 📚 **Tech Stack**

| Category   | Tools             |
| ---------- | ----------------- |
| OCR        | Tesseract, OpenCV |
| AI/LLM     | Ollama, Llama 3.2 |
| Backend    | FastAPI, Uvicorn  |
| Validation | difflib, Pandas   |
| Deployment | Docker, Cloud Run |
| Languages  | Python 3.12       |

---

# 👨‍💻 **Author**

**Konda Charana**
Python Developer & AI/ML Engineer
🔗 LinkedIn: *Konda Charana*
🔗 GitHub: *KondaCharana*

---
