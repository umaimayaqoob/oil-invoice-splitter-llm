# Oil Invoice Splitter using LLM (phi3:mini)

A Python-based tool to process scanned oil trading documents (PDFs). It uses OCR and a local LLM to extract, classify, and split documents like invoices, bills of lading, and certificates.

---

## What It Does

- Converts scanned PDFs to OCR-readable text  
- Uses a local LLM (`phi3:mini` via Ollama) to:
  - Classify each page type  
  - Extract important fields (like invoice number, date, total amount, etc.)
- Splits the original multi-page PDF into individual files for each document

---

## Tools & Technologies

| Tool              | Purpose                                  |
|-------------------|------------------------------------------|
| Tesseract OCR     | Text extraction from scanned pages       |
| Ollama + phi3:mini| Local LLM for classification & extraction|
| PyMuPDF (fitz)    | PDF splitting and page selection         |
| Python            | Entire pipeline scripting                |

---

## Pipeline Steps

1. Extract OCR text → `ocr_pages.json`  
2. Group pages into documents → `chunks.json`  
3. Use LLM to extract fields → `parsed_chunks.json`  
4. Split original PDF into individual files → Saved to `split_pdfs/`

---

## How to Use

### 1. Clone the Repo
```bash
git clone https://github.com/umaimayaqoob/oil-invoice-splitter-llm.git
cd oil-invoice-splitter-llm
2. Set Up Python Environment
bash
Copy
Edit
pip install -r requirements.txt
3. Install and Run Ollama
Download Ollama: https://ollama.com/download

Run this command to get the model:

bash
Copy
Edit
ollama pull phi3:mini
Running the Pipeline
Step 1: OCR & Chunking
Already done manually → ocr_pages.json and chunks.json are available

Step 2: Use LLM to Extract Fields

bash
Copy
Edit
python llm_parser.py
Step 3: Split PDF into Separate Files

bash
Copy
Edit
python split_pdf.py
Output Files
parsed_chunks.json: Extracted fields using LLM

split_pdfs/: Individual document PDFs such as:

invoice_1.pdf

bol_4.pdf

certificate_9.pdf

Notes
Some chunks may fail to parse if OCR text is unclear. Those are skipped.

The project uses phi3:mini model served locally via Ollama.

Once installed, everything runs locally with no cloud dependency.

Author
Umaima Yaqoob
GitHub: umaimayaqoob