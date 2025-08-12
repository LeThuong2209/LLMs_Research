import csv
import os
import fitz
import requests
import subprocess
import re
from pathlib import Path
from typing import List
from PIL import Image
import json
import pytesseract
from pdf2image import convert_from_path
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from crawl4ai import AsyncWebCrawler
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

OLLAMA_MODEL = 'mistral'
OLLAMA_TIMEOUT = 300 

# =====Agent 1: Extractor Agent=====
def create_extractor_agent(model_name="mistral"):
    llm = ChatOllama(model=model_name)

    extract_prompt = PromptTemplate.from_template("""
    You are an AI specialized in analyzing academic papers.

    Your task:
    Read the given text (from 1 page of a research paper) and output **EXACTLY one line** in TSV format with **8 columns in this exact order**:

    Title\tVariables\tTheories\tHypotheses\tMethodology\tDataset(s)\tResults\tLimitation

    ===== RULES =====
    1. Output **only one single line** with 8 columns, separated by **tab characters ('\t')**.
    2. If a field is not found, write exactly: **Not Found** (do NOT leave blank).
    3. If multiple values exist in ONE COLUMN, separate them with **semicolon (';')** — no extra spaces before or after the semicolon.
    4. Do NOT include quotes, explanations, bullet points, or headers — just the raw TSV row.
    5. Column meanings:
       - **Title**: Full paper title (or main topic if partial title found).
       - **Variables**: All studied variables or factors (independent, dependent, control, etc.).
       - **Theories**: Theories, models, or conceptual frameworks used.
       - **Hypotheses**: Research hypotheses or predictions stated in the text.
       - **Methodology**: Research design/method (e.g., survey, experiment, case study).
       - **Dataset(s)**: Information about the dataset — sample size, source, demographic, etc.
       - **Results**: Main findings, correlations, or conclusions.
       - **Limitation**: Study limitations, weaknesses, or constraints.
    6. **Do NOT add extra tabs or columns** — exactly 7 tabs in the output.

    ===== Example Input =====
    "Employee Satisfaction and Productivity
    This study explores the relationship between employee satisfaction and productivity.
    Using Herzberg's Two-Factor Theory and A Theory, we hypothesize that increased job satisfaction
    positively impacts productivity. Data were collected from 200 employees in the IT sector
    using a structured questionnaire. The findings show a strong correlation between satisfaction
    and productivity, but limitations include a small sample size and focus only on one sector."

    ===== Example Correct Output =====
    "Employee Satisfaction and Productivity\tEmployee satisfaction;Productivity\tHerzberg's Two-Factor Theory; A Theory\tJob satisfaction positively impacts productivity\tSurvey research\t200 employees in IT sector\tStrong correlation found\tSmall sample size;Only one sector"
                                                  
    ===== Now process this text and produce ONLY the TSV line: =====
    {text}
    """)

    def extractor(page_text: str) -> str:
        prompt = extract_prompt.format(text=page_text)
        response = llm.invoke(prompt).content.strip()
        return response

    return extractor


# =====Agent 2: Aggregator Agent=====
def create_aggregator_agent(model_name="mistral"):
    llm = ChatOllama(model=model_name)

    aggregate_prompt = PromptTemplate.from_template("""
    You are an AI that aggregates multiple TSV rows extracted from different pages of the same research paper.

    Your task:
    Merge the given rows into **ONE single TSV line** with the same **8 columns in this exact order**:

    Title\tVariables\tTheories\tHypotheses\tMethodology\tDataset(s)\tResults\tLimitation

    ===== RULES =====
    1. Output **only one single line** with 8 columns, separated by *tab characters ('\t')**.
    2. The **Title** column: Always use exactly "{title_hint}" if provided (even if other rows have different titles).
    3. For each column:
       - Merge all unique, non-empty values found across rows.
       - Separate multiple values with **semicolon (';')** — no extra spaces before/after the semicolon.
       - Remove duplicates but keep different phrasings.
    4. If nothing found for a column, write exactly: **Not Found**.
    5. Do NOT include quotes, explanations, bullet points, or headers — just the raw TSV row.
    6. **Do NOT add extra tabs or columns** — exactly 7 tabs in the output.
    7. DO NOT include "Here is the output based on the rules you provided: " or something like that.
    8. If a column contain "Not Found" and the different answer, prefer the latter.
                                                    
    REMEMBER: separating columns by tabs('\t')
                                                    
    ===== Example Input =====
    "Employee Satisfaction and Productivity\tEmployee satisfaction\tHerzberg's Two-Factor Theory\tJob satisfaction positively impacts productivity\tSurvey research\tStrong correlation found\tSmall sample size."
    "Employee Satisfaction and Productivity\tHerzberg's Two-Factor Theory\tStructured interviews\t200 employees in IT sector\tPositive relationship observed\tOnly one sector"

    ===== Example Correct Output =====
    "Employee Satisfaction and Productivity\tEmployee satisfaction\tHerzberg's Two-Factor Theory\tJob satisfaction positively impacts productivity\tSurvey research;Structured interviews\t200 employees in IT sector\tStrong correlation found;Positive relationship observed\tSmall sample size;Only one sector"
                                                    
    ===== Rows to process =====
    {rows}
    """)

    def aggregator(tsv_rows: List[str], title_hint: str = "") -> str:
        prompt = aggregate_prompt.format(rows="\n".join(tsv_rows), title_hint=title_hint)
        return llm.invoke(prompt).content.strip()

    return aggregator

def get_urls(key_word, pages = 3):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    try:
        driver.get("https://scholar.google.com/")
        search_bar = driver.find_element(By.NAME, 'q')
        search_bar.send_keys(key_word)
        search_bar.send_keys(Keys.RETURN)
        input("Solving Captcha, then push 'Enter'...")
        time.sleep(3)
        urls = []
        page = 0
        while True:
            time.sleep(random.uniform(2, 5))
            urlLinks = driver.find_elements(By.XPATH, '//h3[@class="gs_rt"]/a')
            page = page + 1

            for link in urlLinks:
                href = link.get_attribute("href")
                if href:
                    urls.append(href)

            if page >= pages:
                print(f"Reached max pages ({pages}). Stopping.")
                break
            
            #Find button
            try:
                button = driver.find_element(By.LINK_TEXT, "Next")
            except:
                try:
                    button = driver.find_element(By.LINK_TEXT, "Tiếp")
                except:
                    break
            
            if button:
                button.click()
                time.sleep(random.uniform(3, 6))  
            else:
                print("No next button found. Stopping.")
                break

        return urls
    finally:
        driver.quit()

async def get_pdf(urls):
    async with AsyncWebCrawler() as crawler:
        list1 = []
        for i in urls:
            if i.lower().endswith(".pdf"):
                list1.append(i)
                continue
            result = await crawler.arun(i)
            link = None
            soup = BeautifulSoup(result.html, "html.parser")
            for j in soup.find_all("a", href = True):
                href = j["href"]
                x = href.lower()
                if (x.find("/doi/reader") != -1) or (x.find("/pdf") != -1) or (x.find(".pdf") != -1):  
                    link = urljoin(i, href)
                    break
            if link != None:
                list1.append(link)
        return list1
    
def download_pdf(pdf_link, output_dir):
    try:
        file_name = os.path.basename(pdf_link.split('?')[0])
        if not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"

        file_path = os.path.join(output_dir, file_name)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/115.0 Safari/537.36',
                    'Referer': 'https://dl.acm.org/',
                    'Accept': 'application/pdf'}

        r = requests.get(pdf_link, headers=headers, timeout=30)
        r.raise_for_status()

        with open(file_path, 'wb') as f:
            f.write(r.content)
        print(f"Successfully Downloaded: {file_path}")
    except Exception as e:
        print(f"Fail to download {pdf_link}: {e}")

def is_valid_response(response: str) -> bool:
    if not response:
        return False
    parts = response.strip().split("\t")
    return len(parts) == 8

def extract_page_text(pdf_path, page_number):
    """
    Chooses between digital extraction and OCR for a page.
    """
    # Attempt digital extraction first
    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc.load_page(page_number)
            digital_text = page.get_text().strip()
    except Exception as e:
        print(f"[x] Error extracting digital text from page {page_number + 1}: {e}")
        return ""

    # If digital text is sparse, assume it's a scanned page and run OCR
    if len(digital_text) < 200:
        print(f"[**] Page {page_number + 1}: Digital text is sparse. Running OCR as a fallback...")
        try:
            images = convert_from_path(str(pdf_path), first_page=page_number + 1, last_page=page_number + 1, dpi=300)
            if not images:
                print(f"[x] pdf2image conversion failed for page {page_number + 1}.")
                return ""

            print(f"[...] Running Tesseract OCR on page {page_number + 1}...")
            start_time = time.time()
            ocr_text = pytesseract.image_to_string(images[0], lang='eng').strip()
            print(f"[**] Tesseract OCR for page {page_number + 1} finished in {time.time() - start_time:.2f}s.")
            # Return the more detailed text (OCR or digital)
            return ocr_text if len(ocr_text) > len(digital_text) else digital_text
        except Exception as e:
            print(f"[x] Error during OCR for page {page_number + 1}: {e}. Is poppler-utils installed?")
            return digital_text  # Return whatever digital text was found
    else:
        print(f"[**] Page {page_number + 1}: Digital text detected (length: {len(digital_text)}).")
        return digital_text
    
def is_important_page(page_text: str) -> bool:
    keywords = [
        r"\babstract\b", r"\bintroduction\b",
        r"\bmethod(?:ology|s|ological)?\b",  # match method, methods, methodology
        r"\bmaterials?\s+and\s+methods\b",
        r"\bresults?\b", r"\bfindings?\b",
        r"\bdiscussion\b",
        r"\bconclusions?\b", r"\bsummary\b",
        r"\blimitations?\b", r"\bfuture\s+work\b"
    ]
    if len(page_text) < 100:
        return False
    if page_text.strip() == "":
        return False
    if page_text.lower().startswith("references"):
        return False
    for kw in keywords:
        if re.search(kw, page_text, flags=re.IGNORECASE):
            return True
    return False

def extracted(pdf_path: Path):
    extractor_agent = create_extractor_agent(OLLAMA_MODEL)
    aggregator_agent = create_aggregator_agent(OLLAMA_MODEL)

    page_infos = []
    main_title = ""
    # Mở PDF và quét trang
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    print(f"\n---PDF has {num_pages} pages---")
    for page in range(num_pages):
        print(f"--Page {page + 1}/{num_pages}--")
        page_text = extract_page_text(pdf_path, page)
        page_text_lower = page_text.lower()
        if is_important_page(page_text_lower):
            clean_text = " ".join(page_text.split())
            tsv_line = extractor_agent(clean_text)

            if is_valid_response(tsv_line):
                page_infos.append(tsv_line)
                if page == 0:
                    parts = tsv_line.split("\t")
                    if parts and parts[0].strip() != "Not Found":
                        main_title = parts[0].strip()
                print("[->] Valid infor extracted")
        else:
            print("Skip...")
    doc.close()

    if not page_infos:
        return ""

    # Agent 2 tổng hợp
    final_row = aggregator_agent(page_infos, main_title)
    return final_row
            

def main():
    """Main function to parse arguments and drive the processing."""
    key_word = input("Enter the keyword: ")
    pages = int(input("Enter how many pages you want to get (3 is the default): "))
    output_dir = './pdf_save'
    os.makedirs(output_dir, exist_ok=True)

    urls = get_urls(key_word, pages)
    num_url = len(urls)

    print(f"Find {num_url} results: ")
    for url in urls:
        print(url)
        
    pdf_links = asyncio.run(get_pdf(urls))
    #save pdf links in a json file  
    with open("pdf_links.json", "w", encoding="utf-8") as f:
        json.dump(pdf_links, f, ensure_ascii=False, indent=2)

    print(f"\nFound {len(pdf_links)} PDF links:")
    for link in pdf_links:
        download_pdf(link, output_dir)

    headers = ["Title", "Variables", "Theories", "Hypotheses",
            "Methodology", "Dataset(s)", "Results", "Limitation"]

    with open("paper.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)

    for pdf_file in Path(output_dir).glob("*.pdf"):
        row = extracted(pdf_file)
        row = row.strip()
        if not row:
            continue
        row = row.replace("\r", " ").replace("\n", " ")
        row = row.replace('"', '')
        with open("paper.tsv", "a", encoding="utf-8", newline="") as f:
            f.write(row + '\n')

if __name__ == '__main__':
    main()