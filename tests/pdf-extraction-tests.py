import os
import sys
from pypdf import PdfReader

def test_pdf_direct(pdf_path):
    """
    Test PDF extraction directly without mocking or complex setup.
    This will show exactly what text is being extracted from the PDF.
    """
    print(f"Testing direct PDF extraction on: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file does not exist at {pdf_path}")
        return
    
    try:
        # Open the PDF and get basic info
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF has {num_pages} pages")
        
        # Extract text from each page
        total_text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            chars = len(page_text) if page_text else 0
            has_text = chars > 0
            
            print(f"Page {i+1}: {chars} chars, Has text: {has_text}")
            
            if has_text:
                # Save the first 3 pages to see content
                if i < 3:
                    print(f"\n--- Sample from page {i+1} ---")
                    print(page_text[:300] + "..." if len(page_text) > 300 else page_text)
                    print("---\n")
                
                total_text += page_text
        
        print(f"\nTotal extracted text: {len(total_text)} characters")
        
        # Save full text to file for inspection
        output_file = os.path.join(os.path.dirname(pdf_path), "extracted_text.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(total_text)
        
        print(f"Full text saved to: {output_file}")
        
        # Also save first 3 pages to separate files for detailed inspection
        for i in range(min(3, num_pages)):
            page_text = reader.pages[i].extract_text()
            if page_text:
                page_file = os.path.join(os.path.dirname(pdf_path), f"page_{i+1}_text.txt")
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_text)
                print(f"Page {i+1} text saved to: {page_file}")
        
    except Exception as e:
        print(f"Error during PDF extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pdf_path = r"Vaccaro_2024.pdf"
    test_pdf_direct(pdf_path)
