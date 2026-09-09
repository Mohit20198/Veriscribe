import sys
import os
import json
import base64
import logging
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

from ingestion.pipeline import parse_document
from extraction.extractor import extract_facts_from_chunk
from extraction.prompts import CHART_EXTRACTION_PROMPT
from extraction.extractor import ExtractionResult

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    pdf_path = "output/arxiv_paper.pdf"
    
    chunks = parse_document(pdf_path, document_id="arxiv_doc")
    
    chart_chunk = next((c for c in chunks if c.chunk_type == "chart_candidate" and "Performance of Mistral 7B" in (c.text or "")), None)
    if not chart_chunk:
        print("Could not find Figure 4 chart chunk")
        sys.exit(1)
        
    print(f"Testing chart extraction for {chart_chunk.chunk_id}")
    
    with open(chart_chunk.image_path, "rb") as f:
        img_data = f.read()
        
    with open("output/debug_figure4.png", "wb") as f:
        f.write(img_data)
        
    print(f"Saved base64 image to output/debug_figure4.png")
    
    img_b64 = base64.b64encode(img_data).decode("utf-8")
    
    content = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{img_b64}"
        }
    }]
    if chart_chunk.text:
        content.append({"type": "text", "text": f"Chart Caption: {chart_chunk.text}"})
        
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
        {"role": "system", "content": CHART_EXTRACTION_PROMPT + "\n\nCRITICAL: You MUST extract facts for ALL data points. This is a complex chart with 32 data points. Take your time, think step-by-step in the message content first, and then call the submit_extracted_facts tool."},
        {"role": "user", "content": content}
    ]
    
    # Test 3: response_format JSON schema
    print("\n--- TEST 3: response_format ---")
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-5",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ExtractionResult",
                    "schema": ExtractionResult.model_json_schema(),
                    "strict": True
                }
            },
            max_tokens=4000
        )
        
        msg = response.choices[0].message
        print(f"Message content: {msg.content}")
    except Exception as e:
        print(f"Error: {e}")
        


if __name__ == "__main__":
    main()
