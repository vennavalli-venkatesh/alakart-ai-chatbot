import re
from typing import List

from app.services.document_loader import Document


class DataCleaner:
    """
    Cleans and normalizes document text.
    
    Goals:
    - Remove null/empty text
    - Normalize excessive whitespace
    - Normalize repeated blank lines
    - Remove obvious PDF extraction artifacts
    - Clean OCR noise (stray characters, garbled fragments)
    - Preserve headings, bullet points, medical terminology, dosage information.
    """

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        import json
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                data = json.loads(json_str)
                if isinstance(data, list):
                    # It's a valid JSON array, do not corrupt it with OCR cleaning
                    return text
        except Exception:
            pass

        # -----------------------------------------
        # OCR-specific cleaning
        # -----------------------------------------

        # Remove stray single special characters
        # on their own line (common OCR noise)
        text = re.sub(
            r'^\s*[|~`^\\]{1,2}\s*$',
            '',
            text,
            flags=re.MULTILINE
        )

        # Remove lines that are only punctuation
        # or non-alphanumeric noise
        text = re.sub(
            r'^\s*[^\w\s]{1,3}\s*$',
            '',
            text,
            flags=re.MULTILINE
        )

        # Normalize common OCR ligature/encoding
        # artifacts
        text = text.replace('\u2019', "'")
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '-')
        text = text.replace('\u2022', '-')

        # -----------------------------------------
        # General cleaning (preserved from original)
        # -----------------------------------------

        # Replace 3+ newlines with 2 newlines.
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean lines but preserve intended newlines
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            cleaned_lines = []
            for line in lines:
                line = re.sub(r'[ \t]+', ' ', line).strip()
                if line:
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                # Rejoin lines in a paragraph with a single space (or newline if it's a list)
                # If a line starts with a bullet point like -, *, or a number, keep the newline
                merged_para = ""
                for i, line in enumerate(cleaned_lines):
                    if i == 0:
                        merged_para += line
                    else:
                        # Check if line looks like a bullet or numbered list
                        if re.match(r'^[-*]|^\d+\.', line):
                            merged_para += "\n" + line
                        else:
                            # Join with space
                            merged_para += " " + line
                
                cleaned_paragraphs.append(merged_para)
                
        return "\n\n".join(cleaned_paragraphs)

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Cleans a list of Document objects.
        Original metadata is preserved.
        """
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            if cleaned_content:
                cleaned_docs.append(
                    Document(page_content=cleaned_content, metadata=doc.metadata)
                )
        return cleaned_docs
