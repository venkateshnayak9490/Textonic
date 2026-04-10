import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from pdf_extractor import PDFExtractor
from text_filter import TextFilter
from chunker import TextChunker
from config import CHUNK_SIZE, CHUNK_OVERLAP, SCORE_THRESHOLD


class Stage1Pipeline:
    def __init__(
        self,
        score_threshold: int = SCORE_THRESHOLD,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.extractor = PDFExtractor()
        self.filter = TextFilter(score_threshold=score_threshold)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        
        self.score_threshold = score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdfs(
        self, 
        pdf_directory: str, 
        output_file: str = 'chunks.json',
        verbose: bool = True
    ) -> Dict:
        if verbose:
            print("="*70)
            print("STAGE 1: DATA PREPARATION PIPELINE")
            print("="*70)
            print()
        
        if verbose:
            print("STEP 1: Extracting text from PDFs")
            print("-"*70)
        
        extraction_results = self.extractor.extract_from_directory(pdf_directory)
        
        if not extraction_results:
            print("ERROR: No PDFs extracted successfully")
            return {'status': 'error', 'message': 'No PDFs extracted'}
        
        successful_extractions = [
            r for r in extraction_results 
            if r['extraction_status'] in ['success', 'warning']
        ]
        
        if verbose:
            print()
        
        if verbose:
            print("STEP 2: Filtering text (content-based)")
            print("-"*70)
        
        filtered_documents = []
        
        for i, extraction in enumerate(successful_extractions, 1):
            if verbose:
                print(f"\nFiltering document {i}/{len(successful_extractions)}: {extraction['filename']}")
            
            filter_result = self.filter.filter_text(
                extraction['extracted_text'],
                verbose=verbose
            )
            
            filtered_documents.append({
                'source_id': Path(extraction['filename']).stem,  # filename without extension
                'filename': extraction['filename'],
                'filtered_text': filter_result['filtered_text'],
                'statistics': filter_result['statistics']
            })
        
        if verbose:
            print()
        
        if verbose:
            print("STEP 3: Creating chunks")
            print("-"*70)
        
        chunk_result = self.chunker.chunk_multiple_documents(filtered_documents)
        chunks = chunk_result['chunks']
        chunk_stats = chunk_result['statistics']
        
        if verbose:
            print(f"Created {chunk_stats['total_chunks']} chunks from {chunk_stats['total_documents']} documents")
            print(f"Average: {chunk_stats['avg_chunks_per_doc']:.1f} chunks per document")
            print(f"Average chunk length: {chunk_stats['avg_chunk_length']:.0f} characters")
            print(f"Average word count: {chunk_stats['avg_word_count']:.1f} words")
            print()
        
        if verbose:
            print("STEP 4: Saving results")
            print("-"*70)
        
        # Prepare output data
        output_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_pdfs': len(extraction_results),
                'successful_pdfs': len(successful_extractions),
                'total_chunks': len(chunks),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'score_threshold': self.score_threshold,
                'avg_chunks_per_pdf': chunk_stats['avg_chunks_per_doc'],
                'pipeline_statistics': {
                    'extraction': {
                        'total_pdfs': len(extraction_results),
                        'successful': len(successful_extractions),
                        'failed': len(extraction_results) - len(successful_extractions)
                    },
                    'filtering': {
                        'avg_retention_rate': sum(d['statistics']['retention_rate'] for d in filtered_documents) / len(filtered_documents) if filtered_documents else 0,
                        'avg_keyword_density': sum(d['statistics']['avg_keyword_density'] for d in filtered_documents) / len(filtered_documents) if filtered_documents else 0
                    },
                    'chunking': chunk_stats
                }
            },
            'chunks': chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"✓ Saved {len(chunks)} chunks to {output_file}")
            print()
        
        if verbose:
            print("="*70)
            print("PIPELINE COMPLETE - SUMMARY")
            print("="*70)
            print(f"Input PDFs: {len(extraction_results)}")
            print(f"Successfully processed: {len(successful_extractions)}")
            print(f"Total chunks created: {len(chunks)}")
            print(f"Output file: {output_file}")
            print(f"Average retention rate: {output_data['metadata']['pipeline_statistics']['filtering']['avg_retention_rate']:.1%}")
            print("="*70)
        
        return {
            'status': 'success',
            'output_file': output_file,
            'metadata': output_data['metadata'],
            'chunks': chunks
        }


# Main execution
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        pdf_dir = sys.argv[1]
    else:
        pdf_dir = '/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/pdfs'  # Default directory
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = '/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/chunks.json'
    
    pipeline = Stage1Pipeline()
    result = pipeline.process_pdfs(
        pdf_directory=pdf_dir,
        output_file=output_file,
        verbose=True
    )
    
    if result['status'] == 'success':
        print(f"\n Success! Chunks saved to {result['output_file']}")
    else:
        print(f"\n Pipeline failed: {result.get('message', 'Unknown error')}")