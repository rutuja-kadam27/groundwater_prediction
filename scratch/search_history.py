# scratch/search_history.py
import json

transcript_path = r"C:\Users\VARAD\.gemini\antigravity\brain\4bcd018b-c689-4fbb-8cec-25c18e08bc65\.system_generated\logs\transcript.jsonl"

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            step = json.loads(line)
            if step.get("step_index") == 892:
                print(json.dumps(step.get("tool_calls"), indent=2))
        except Exception as e:
            pass
