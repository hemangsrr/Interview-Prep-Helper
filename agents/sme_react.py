from __future__ import annotations
from typing import Dict, TypedDict, List, Callable
from llm import LLM

# State definition shared by orchestrator and SMEs
class InterviewState(TypedDict, total=False):
    prompts: List[Dict]  # list of {name, system_prompt}
    resume_notes: str
    history: List[Dict]  # {agent, question, user_answer, feedback}
    turn_index: int
    max_turns: int
    current_agent: str
    mode: str  # 'route' | 'ask' | 'await_answer' | 'feedback' | 'done'
    last_question: str
    last_feedback: str
    user_answer: str
    force_stop: bool


def build_sme_node(system_prompt: str, llm: LLM) -> Callable[[InterviewState], InterviewState]:
    """
    Return a LangGraph-compatible node function for an SME agent.
    It reads state['mode'] and either:
      - when 'ask': produces a question based on resume_notes + recent history
      - when 'feedback': produces feedback based on last question and user_answer
    """

    def node(state: InterviewState) -> InterviewState:
        mode = state.get("mode")
        name = state.get("current_agent", "Expert")
        if mode == "ask":
            context = []
            if state.get("resume_notes"):
                context.append("Resume Notes:\n" + state["resume_notes"]) 
            recent = state.get("history", [])[-5:]
            if recent:
                context.append("History (most recent last):\n" + "\n".join([
                    f"Q by {h.get('agent')}: {h.get('question')}\nUser: {h.get('user_answer','')}" for h in recent
                ]))
            ask_instr = (
                "Craft the next interview question. Ask a single, clear, challenging question tailored to the role and context."
            )
            user = "\n\n".join(context) + "\n\n" + ask_instr
            question = (llm.invoke(system_prompt, user) or "").strip()
            state["last_question"] = question
            state["mode"] = "await_answer"
            return state
        elif mode == "feedback":
            q = state.get("last_question", "")
            a = state.get("user_answer", "")
            fb_instr = "Provide brief feedback in 2-4 bullets: strengths and improvements."
            user = f"Question: {q}\nAnswer: {a}\n\n{fb_instr}"
            feedback = (llm.invoke(system_prompt, user) or "").strip()
            state["last_feedback"] = feedback
            # append to history
            hist = state.get("history", [])
            hist.append({
                "agent": name,
                "question": state.get("last_question", ""),
                "user_answer": a,
                "feedback": feedback,
            })
            state["history"] = hist
            state["mode"] = "route"
            return state
        else:
            return state

    return node
