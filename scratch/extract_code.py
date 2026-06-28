# scratch/extract_code.py
import json

transcript_path = r"C:\Users\VARAD\.gemini\antigravity\brain\4bcd018b-c689-4fbb-8cec-25c18e08bc65\.system_generated\logs\transcript.jsonl"

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            step = json.loads(line)
            if step.get("step_index") == 892:
                tc = step["tool_calls"][0]
                code = tc["args"]["CodeContent"]
                
                # Double-encoded JSON string decoding
                decoded = code
                if isinstance(code, str):
                    if code.startswith('"') and code.endswith('"'):
                        try:
                            decoded = json.loads(code)
                        except Exception:
                            # Fallback: strip outer quotes and decode escape characters
                            inner = code[1:-1]
                            decoded = inner.encode('utf-8').decode('unicode_escape')
                    else:
                        try:
                            decoded = code.encode('utf-8').decode('unicode_escape')
                        except Exception:
                            decoded = code
                            
                with open("scratch/extracted_ml_models.py", "w", encoding="utf-8") as out:
                    out.write(decoded)
                print("Successfully extracted code to scratch/extracted_ml_models.py!")
        except Exception as e:
            print("Error:", e)
