from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path
import json
from datetime import datetime

class LLMLoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.log_dir = Path("logs/llm_detailed")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_interaction = {}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_interaction["timestamp_start"] = datetime.now().isoformat()
        self.current_interaction["prompts"] = prompts
    
    def on_llm_end(self, response, **kwargs):
        self.current_interaction["timestamp_end"] = datetime.now().isoformat()
        self.current_interaction["response"] = response.generations[0][0].text
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"detailed_interaction_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.current_interaction, f, indent=2, ensure_ascii=False)
        
        self.current_interaction = {} 