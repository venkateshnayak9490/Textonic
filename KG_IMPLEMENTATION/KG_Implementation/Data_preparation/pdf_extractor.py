"""
PDF text extraction using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List
import os


class PDFExtractor:

    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:

        result = {
            'filename': os.path.basename(pdf_path),
            'filepath': pdf_path,
            'total_pages': 0,
            'extracted_text': '',
            'text_length': 0,
            'extraction_status': 'success',
            'error_message': None
        }
        
        try:
            doc = fitz.open(pdf_path)
            result['total_pages'] = len(doc)
            
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text_parts.append(page_text)
            
            full_text = '\n\n'.join(text_parts)
            
            result['extracted_text'] = full_text
            result['text_length'] = len(full_text)
            
            doc.close()
            
            if result['text_length'] < 100:
                result['extraction_status'] = 'warning'
                result['error_message'] = 'Very short text extracted - may be scanned PDF'
            
        except FileNotFoundError:
            result['extraction_status'] = 'error'
            result['error_message'] = 'PDF file not found'
            
        except Exception as e:
            result['extraction_status'] = 'error'
            result['error_message'] = f'Extraction failed: {str(e)}'
        
        return result
    
    def extract_from_directory(self, directory: str) -> List[Dict]:

        results = []
        
        # Get all PDF files
        pdf_files = list(Path(directory).glob('*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return results
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"Extracting {i}/{len(pdf_files)}: {pdf_path.name}")
            
            result = self.extract_text_from_pdf(str(pdf_path))
            results.append(result)
            
            if result['extraction_status'] == 'success':
                print(f"  Extracted {result['text_length']} characters from {result['total_pages']} pages")
            elif result['extraction_status'] == 'warning':
                print(f"  Warning: {result['error_message']}")
            else:
                print(f"  Error: {result['error_message']}")
        
        successful = sum(1 for r in results if r['extraction_status'] == 'success')
        print(f"\nExtraction complete: {successful}/{len(pdf_files)} successful")
        
        return results


# Example usage
if __name__ == '__main__':
    extractor = PDFExtractor()
    
    # Extract from single PDF
    result = extractor.extract_text_from_pdf('example.pdf')
    print(f"Status: {result['extraction_status']}")
    print(f"Text length: {result['text_length']}")
    
    # Or extract from directory
    # results = extractor.extract_from_directory('path/to/pdfs/')