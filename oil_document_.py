
import os
import json
import re
import logging
import fitz
import ollama
import pytesseract
from PIL import Image
from time import sleep
from tqdm import tqdm
from typing import List, Dict
from pdf2image import convert_from_path


class OilDocumentProcessor:
    def __init__(self,
                 pdf_path: str,
                 poppler_path: str,
                 tesseract_cmd: str,
                 base_dir: str,
                 dpi: int = 300,
                 ollama_model: str = "phi3:mini"):
        self.pdf_path = pdf_path
        self.poppler_path = poppler_path
        self.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self.model_name = ollama_model

        self.image_dir = os.path.join(base_dir, "pages")
        self.ocr_text_dir = os.path.join(base_dir, "ocr_text")
        self.ocr_json_path = os.path.join(base_dir, "ocr_pages.json")
        self.chunks_path = os.path.join(base_dir, "chunks.json")
        self.parsed_chunks_path = os.path.join(base_dir, "parsed_chunks.json")
        self.failed_log_path = os.path.join(base_dir, "failed_chunks_log.txt")
        self.output_split_pdf_dir = os.path.join(base_dir, "split_pdfs")

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.ocr_text_dir, exist_ok=True)
        os.makedirs(self.output_split_pdf_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.failed_log_path), exist_ok=True)

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("OilDocumentProcessor")

    def convert_pdf_to_images(self) -> None:
        images = convert_from_path(self.pdf_path, dpi=self.dpi, poppler_path=self.poppler_path)
        for i, image in enumerate(images):
            filename = os.path.join(self.image_dir, f"page_{i+1:03}.png")
            try:
                image.save(filename, "PNG")
                self.logger.info(f"Saved: {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save {filename}: {e}")

    def perform_ocr(self) -> None:
        ocr_results = []
        page_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])

        for fname in tqdm(page_files, desc="OCR Processing"):
            page_num = int(fname.split("_")[1].split(".")[0])
            image_path = os.path.join(self.image_dir, fname)
            image = Image.open(image_path).rotate(270, expand=True)
            config = "--psm 6 --oem 3 -l eng"
            text = pytesseract.image_to_string(image, config=config).strip()

            txt_file_path = os.path.join(self.ocr_text_dir, f"page_{page_num:03}.txt")
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(text)

            ocr_results.append({
                "page": page_num,
                "filename": fname,
                "text": text,
                "is_blank": len(text) < 50
            })

        with open(self.ocr_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=2)

        self.logger.info(f"OCR complete for {len(page_files)} pages")

    def chunk_documents(self) -> None:
        with open(self.ocr_json_path, "r", encoding="utf-8") as f:
            pages = sorted(json.load(f), key=lambda x: x["page"])

        chunks = []
        i = 0
        chunk_id = 1

        while i < len(pages):
            page_num = pages[i]['page']
            is_blank = pages[i]['is_blank']

            if page_num <= 24:
                chunk_pages = pages[i:i+3]
                combined_text = "\n".join(p['text'] for p in chunk_pages)
                chunks.append({
                    "chunk_id": chunk_id,
                    "type_hint": "invoice",
                    "pages": [p['page'] for p in chunk_pages],
                    "text": combined_text
                })
                chunk_id += 1
                i += 4
            elif 25 <= page_num <= 38:
                chunks.append({
                    "chunk_id": chunk_id,
                    "type_hint": "bol",
                    "pages": [pages[i]['page']],
                    "text": pages[i]['text']
                })
                chunk_id += 1
                i += 1
            elif page_num >= 39:
                if not is_blank:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "type_hint": "certificate",
                        "pages": [pages[i]['page']],
                        "text": pages[i]['text']
                    })
                    chunk_id += 1
                    i += 2
                else:
                    i += 1

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        self.logger.info(f"Chunking complete. Total chunks: {len(chunks)}")

    def extract_json(self, raw_text: str) -> Dict:
        try:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            return json.loads(match.group())
        except Exception as e:
            self.logger.warning(f"JSON parse error: {e}")
            return None

    def parse_chunks_with_ollama(self) -> None:
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        parsed_chunks = []

        for chunk in chunks:
            text = chunk["text"].strip()
            if len(text) < 50:
                continue

            prompt = f'''
You are an intelligent document parser for oil trading documents.

Analyze the following document text and extract the following fields. Respond ONLY with a valid JSON object, no explanations.

Document text:
"""{text}"""

JSON format:
{{
  "document_type": "Invoice | Bill of Lading | Certificate",
  "invoice_number": "",
  "issue_date": "",
  "due_date": "",
  "buyer": "",
  "seller": "",
  "total_amount_usd": "",
  "vessel_name": "",
  "bbl_quantity": "",
  "bl_date": "",
  "port_of_loading": "",
  "port_of_discharge": "",
  "suggested_filename": ""
}}
'''

            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = self.extract_json(response['message']['content'])
                if not result:
                    raise ValueError("Could not extract valid JSON")

                result["chunk_id"] = chunk["chunk_id"]
                result["pages"] = chunk["pages"]
                parsed_chunks.append(result)
                self.logger.info(f"Parsed chunk {chunk['chunk_id']}")
                sleep(1)

            except Exception as e:
                self.logger.error(f"Failed chunk {chunk['chunk_id']}: {e}")
                with open(self.failed_log_path, "a", encoding="utf-8") as log:
                    log.write(f"Chunk {chunk['chunk_id']} failed: {e}\n\n")
                continue

        with open(self.parsed_chunks_path, "w", encoding="utf-8") as f:
            json.dump(parsed_chunks, f, indent=2)

        self.logger.info("Parsed chunks saved. Model used: " + self.model_name)

    def split_pdf_by_chunks(self) -> None:
        def sanitize_filename(name: str) -> str:
            name = name.strip().replace(" ", "_")
            return re.sub(r'[\\/*?:"<>|]', "", name) or "document"

        with open(self.parsed_chunks_path, "r", encoding="utf-8") as f:
            parsed_chunks = json.load(f)

        doc = fitz.open(self.pdf_path)

        for chunk in parsed_chunks:
            pages = chunk["pages"]
            doc_type = (chunk.get("document_type") or "unknown").lower().replace(" ", "_")
            raw_filename = chunk.get("suggested_filename") or f"{doc_type}_{chunk['chunk_id']}"
            filename = sanitize_filename(raw_filename)
            if not filename.endswith(".pdf"):
                filename += ".pdf"

            output_path = os.path.join(self.output_split_pdf_dir, filename)
            new_doc = fitz.open()

            for page_num in pages:
                new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)

            try:
                new_doc.save(output_path)
                self.logger.info(f"Saved: {filename} ({len(pages)} pages)")
            except Exception as e:
                self.logger.error(f"Failed to save {filename}: {e}")

            new_doc.close()

        doc.close()
        self.logger.info(f"PDF splitting complete. Files saved in: {self.output_split_pdf_dir}")


def main():
    processor = OilDocumentProcessor(
        pdf_path="D:/LLM/oil-invoice-splitter-llm/data/acartwright_250505-110900-58d.pdf",
        poppler_path="C:/poppler-24.08.0/Library/bin",
        tesseract_cmd="D:/Tesseract-OCR/tesseract.exe",
        base_dir="D:/LLM/oil-invoice-splitter-llm/data",
        ollama_model="phi3:mini"
    )

    processor.convert_pdf_to_images()
    processor.perform_ocr()
    processor.chunk_documents()
    processor.parse_chunks_with_ollama()
    processor.split_pdf_by_chunks()


if __name__ == "__main__":
    main()
