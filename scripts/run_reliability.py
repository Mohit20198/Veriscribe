"""
scripts/run_reliability.py
--------------------------
Reliability test script to verify structured JSON extraction for a specific TEXT_MODEL.
"""

import sys
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from ingestion.pipeline import parse_document
from extraction.extractor import ExtractionResult
from extraction.prompts import TEXT_EXTRACTION_PROMPT

logging.basicConfig(level=logging.INFO)

def test_model_reliability(client, model_name, chunks, context_window=""):
    print(f"\n--- Testing reliability for {model_name} ---")
    
    success_count = 0
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        content = []
        user_text = f"CONTEXT WINDOW:\n{context_window}\n\nTARGET CHUNK:\n{chunk.text}"
        content.append({"type": "text", "text": user_text})
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_extracted_facts",
                    "description": "Submit the list of extracted facts.",
                    "parameters": ExtractionResult.model_json_schema()
                }
            }
        ]
        
        messages = [
            {"role": "system", "content": TEXT_EXTRACTION_PROMPT},
            {"role": "user", "content": content}
        ]
        
        print(f"[{i+1}/{total_chunks}] Sending request...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "submit_extracted_facts"}}
            )
            
            # Check parsing
            valid = False
            if response.choices and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "submit_extracted_facts":
                        args = json.loads(tool_call.function.arguments)
                        result = ExtractionResult(**args)
                        valid = True
                        break
            
            if valid:
                print("  -> Success")
                success_count += 1
            else:
                print("  -> Failed: No valid tool calls found")
                
        except Exception as e:
            print(f"  -> Failed: Exception occurred: {e}")
            
    success_rate = (success_count / total_chunks) * 100
    print(f"\nResults for {model_name}: {success_rate:.1f}% valid/parseable ({success_count}/{total_chunks})")
    return success_rate

def main():
    load_dotenv()
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)
        
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    pdf_path = "output/arxiv_paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        sys.exit(1)
        
    print(f"Parsing {pdf_path} for sample chunks...")
    chunks = parse_document(pdf_path, document_id="demo_doc")
    
    text_chunks = [c for c in chunks if c.chunk_type == "text" and c.text]
    
    # Take 10 sample chunks
    sample_chunks = text_chunks[:10]
    if len(sample_chunks) < 10:
        print(f"Warning: Only found {len(sample_chunks)} text chunks.")
        
    primary_model = "deepseek/deepseek-v4-flash-vision-exp"
    fallback_model = "moonshotai/kimi-k2.6"
    
    rate = test_model_reliability(client, primary_model, sample_chunks)
    
    if rate < 90.0:
        print(f"\n{primary_model} fell below 90% threshold. Falling back and testing {fallback_model}...")
        test_model_reliability(client, fallback_model, sample_chunks)
    else:
        print(f"\n{primary_model} passed the reliability threshold.")

if __name__ == "__main__":
    main()
