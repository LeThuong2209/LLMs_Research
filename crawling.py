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

OLLAMA_MODEL = 'mistral'
OLLAMA_TIMEOUT = 300 

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

def build_prompt(text: str, main_title = ""):
    title_hint = ""
    if main_title:
        title_hint = f"\nThe title of the paper has already been identified as: \"{main_title}\".\nUse this title exactly in the output.\n"

    return f"""You are an AI specialized in analyzing academic papers. 
        Your task is to process one page of a PDF file, read its extracted text, and identify the following information:

        1. The main title of the entire academic paper (NOT a section title or heading in this page):
            - Usually the largest text at the top of the first page, before the abstract/introduction.
            - NEVER use subsection titles such as "Abstract" or "Introduction" as the Title.
            - If not visible in the given text, use the known title or infer from context.{title_hint}
        2. The variables the paper using
        3. The theories used in the paper
        4. The hypotheses used in the paper
        5. The methodology that the paper use to research
        6. The dataset(s) used in the research, if mentioned.
        7. The last findings of the research
        8. The limitation of the research

        Output EXACTLY one line of TSV with 8 columns separated by tabs (\t), no more, no less, in this order:
        Title\tVariables\tTheories\tHypotheses\tMethodology\tDataset(s)\tResults\tLimitation

        NEVER output contains a header row. 
        NEVER output contains the content of the header row, only contains the info.
        DO NOT contains the unrelevant information.

        If any field is not found, write "Not Found" in that column.
        If any column contains many answers, seperating each answer by ';'
        If any field contains a semicolon, enclose it in double quotes.
        Do NOT output any explanation or extra text.

        ONLY find the information that suitable for each collumn.

        The title is usually the main heading of the documents in the FIRST page.

        ### Example Input Text:
        "Employee Satisfaction and Productivity
        This study explores the relationship between employee satisfaction and productivity.
        Using Herzberg's Two-Factor Theory, we hypothesize that increased job satisfaction
        positively impacts productivity. Data were collected from 200 employees in the IT sector
        using a structured questionnaire. The findings show a strong correlation between satisfaction
        and productivity, but limitations include a small sample size and focus only on one sector."

        ### Example Correct Output:
        Employee Satisfaction and Productivity\tEmployee satisfaction;Productivity\tHerzberg's Two-Factor Theory\tJob satisfaction positively impacts productivity\tSurvey research\t200 employees in IT sector\tStrong correlation found\tSmall sample size;Only one sector

        Now, process the following extracted text and produce ONLY the TSV line as per rules above:
        {text}
        """

def is_valid_response(response: str) -> bool:
    if not response:
        return False
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    return any(line.count('\t') >= 7 for line in lines)

def call_ollama(prompt: str, model: str) -> str:
    """Calls the Ollama model locally."""
    print(f"  [->] Calling Ollama model '{model}'...")
    start_time = time.time()
    try:
        process = subprocess.run(
            ["ollama", "run", model],
            input=prompt, capture_output=True, text=True, encoding='utf-8', check=False, timeout=OLLAMA_TIMEOUT
        )
        print(f"  [<-] Ollama call finished in {time.time() - start_time:.2f}s. Exit code: {process.returncode}")
        if process.returncode != 0:
            print(f"  [x] Ollama Error Stderr: {process.stderr.strip()}")
        return process.stdout.strip()
    except FileNotFoundError:
        print("  [x] 'Ollama' command not found. Is Ollama installed and in your PATH?")
        return ""
    except Exception as e:
        print(f"  [x] An unexpected error occurred during the Ollama call: {e}")
        return ""

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
    """
    Process a single PDF, scan important pages, call Ollama, return a list of extracted rows ["title\tvariables\ttheories\thypotheses\tmethodology\tdataset(s)\tresults\tlimitation"].
    """

    page_infos = []
    main_title = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close()
    except Exception as e:
        print(f"[x] CRITICAL: Could not open PDF '{pdf_path.name}'. Skipping. Error: {e}")
        return []
    
    print(f"\n[*] PDF has {num_pages} pages. Scanning all pages and extracting info...")
    for page in range(num_pages):
        print(f"-- Page {page + 1}/{num_pages} --")

        page_text = extract_page_text(pdf_path, page)
        page_text_lower = page_text.lower()
        
        if is_important_page(page_text_lower):
            clean_text = " ".join(page_text.split())
            
            if page == 0:
                prompt_for_title = build_prompt(clean_text)
                for attempt in range(3):
                    response = call_ollama(prompt_for_title, OLLAMA_MODEL)
                    response = response.replace("\r", " ").replace("\n", " ")
                    if is_valid_response(response):
                        print(f"  [Done] Valid information data extracted from page {page + 1} on attempt {attempt + 1}.")
                        # Add the valid rows to our list for this PDF
                        page_infos.append(response.strip())
                        parsed = next(csv.reader([response], delimiter="\t"))
                        if parsed and parsed[0].strip():
                            main_title = parsed[0].strip()
                        break
                    else:
                        print(f"  [Failed] Invalid response on attempt {attempt + 1}/3. Retrying...")
                        if response: 
                            print(f"    (Invalid Response Sample: {response.strip()[:100]}...)")
                        time.sleep(2)
                continue

            if main_title:
                clean_text = f"TITLE_OF_PAPER: {main_title}\n\n{clean_text}"
            prompt = build_prompt(clean_text, main_title)
            for attempt in range(3):
                response = call_ollama(prompt, OLLAMA_MODEL)
                if is_valid_response(response):
                    print(f"  [Done] Valid data extracted from page {page + 1}")
                    page_infos.append(response.strip())
                    break
                else:
                    print(f"  [...] Retry attempt {attempt + 1}/3")
                    time.sleep(2)
            else:
                print("[x] Failed to extract valid info.")
        else:
            print("[->] Skipping calling Ollama")
    if not page_infos:
        return ""
    #call LLMs second time for summarizing
    print("\n[->] Calling Ollama second time for summarizing.")
    aggregate_prompt = f"""
    You are an AI that aggregates extracted TSV rows from different pages of the same academic paper.
    
    You will be given a text, your mission is choosing the best answer for each column.
    To be more specific, the title column must be one answer, the other columns can have more than 1 answer (each answer is seperated by comma)
    DO NOT include "Here are the aggregated rows based on the given text snippets" or something like that in the output.
    
    Each row follows the format (no header row):
    title\tvariables\ttheories\thypotheses\tmethodology\tdataset(s)\tresults\tlimitation

    REMEMBER: just fill the information, not including the name of the column.
    
    Here are multiple extracted rows (TSV format), each from a different page of the PDF:
    {chr(10).join(page_infos)}

    ### Example Input Rows:
    "Employee Satisfaction and Productivity\tEmployee satisfaction\tHerzberg's Two-Factor Theory\tJob satisfaction positively impacts productivity\tSurvey research\tStrong correlation found\tSmall sample size"
    "Employee Satisfaction and Productivity\tHerzberg's Two-Factor Theory\tStructured interviews\t200 employees in IT sector\tPositive relationship observed\tOnly one sector"

    ### Example Correct Aggregated Output:
    "Employee Satisfaction and Productivity\tEmployee satisfaction\tHerzberg's Two-Factor Theory\tJob satisfaction positively impacts productivity\tSurvey research;Structured interviews\t200 employees in IT sector\tStrong correlation found;Positive relationship observed\tSmall sample size;Only one sector"

    Formatting Rules:
    - The Title column MUST NOT be empty(prefer using: {main_title if main_title else '[No title found]'}).
    - Merge duplicate or overlapping data without repeating identical items.
    - Keep the ORDER of columns EXACTLY as:
    title\tvariables\ttheories\thypotheses\tmethodology\tdataset(s)\tresults\tlimitation
    - If a column has multiple distinct values, separate them with semicolons.
    - Columns may be empty if no information was found.
    - Do **NOT** add any explanation or header row.
    - If you cannot find the valid information from whole academic paper for which column, fill in the column: "Not Found".
    """

    final_row = call_ollama(aggregate_prompt, OLLAMA_MODEL)

    return final_row.strip()

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
        for s in headers:
            row = row.replace(s, "")
        parsed = next(csv.reader([row], delimiter='\t'))
        with open("paper.tsv", "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(parsed)

if __name__ == '__main__':
    main()