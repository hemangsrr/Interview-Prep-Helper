from typing import List, Dict
from llm import LLM

JD_PANEL_PROMPT = (
    "You are configuring a mock interview panel based on a given Job Description (JD). "
    "Choose exactly 3 SME roles and write a concise, effective system prompt for each SME to conduct interviews. "
    "Return JSON with an array 'panel' of 3 objects: {name, system_prompt}. Keep prompts under 200 words each."
)


def propose_panel_from_jd(llm: LLM, jd_text: str) -> List[Dict]:
    user = f"JD:\n{jd_text[:4000]}\n\nReturn JSON with array 'panel' of 3 objects as specified."
    content = llm.invoke(JD_PANEL_PROMPT, user, json=True)
    import json
    try:
        data = json.loads(content)
        panel = data.get("panel")
        if isinstance(panel, list) and len(panel) == 3:
            # normalize
            out = []
            for i, p in enumerate(panel):
                out.append({
                    "name": p.get("name") or f"Expert {i+1}",
                    "system_prompt": p.get("system_prompt") or "You are a helpful expert interviewer.",
                })
            return out
    except Exception:
        pass
    # fallback simple panel
    return [
        {"name": "Domain Expert", "system_prompt": "You are a domain expert interviewer."},
        {"name": "Systems Expert", "system_prompt": "You are a systems design interviewer."},
        {"name": "Behavioral Expert", "system_prompt": "You are a behavioral interviewer."},
    ]
