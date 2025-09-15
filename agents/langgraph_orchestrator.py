from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from .sme_react import InterviewState, build_sme_node
from .panel_builder import propose_panel_from_jd
from llm import LLM


@dataclass
class LangGraphOrchestrator:
    llm: LLM
    panel: List[Dict] = field(default_factory=list)
    resume_notes: str = ""
    max_turns: int = 8

    # runtime
    _graph = None
    _app = None
    state: InterviewState = field(default_factory=dict)

    @staticmethod
    def generate_panel(client, jd_text: str) -> List[Dict]:
        """Use LLM to propose 3 SME prompts from the JD."""
        return propose_panel_from_jd(client, jd_text)

    def init_graph(self, panel: List[Dict], resume_notes: str = ""):
        self.panel = panel
        self.resume_notes = resume_notes or ""

        # Build SME nodes with provided prompts
        sme_nodes = []
        for i, p in enumerate(self.panel):
            node = build_sme_node(p.get("system_prompt", "You are an expert interviewer."), self.llm)
            sme_nodes.append((f"sme_{i}", node, p.get("name", f"Expert {i+1}")))

        def route(state: InterviewState) -> InterviewState:
            # stopping conditions
            if state.get("force_stop") or (state.get("turn_index", 0) >= state.get("max_turns", self.max_turns)):
                state["mode"] = "done"
                return state

            mode = state.get("mode")
            if mode in (None, "route"):
                # select next agent in round-robin based on turn_index modulo panel size
                idx = (state.get("turn_index", 0)) % len(self.panel)
                state["current_agent"] = self.panel[idx].get("name", f"Expert {idx+1}")
                state["mode"] = "ask"
                return state
            return state

        def done_node(state: InterviewState) -> InterviewState:
            state["mode"] = "done"
            return state

        builder = StateGraph(InterviewState)
        builder.add_node("route", route)
        for name, node, _label in sme_nodes:
            builder.add_node(name, node)
        builder.add_node("done", done_node)

        # Entry point
        builder.set_entry_point("route")

        # From route, go to specific SME or done
        def from_route_cond(state: InterviewState):
            mode = state.get("mode")
            if mode == "done":
                return "done"
            # pick SME based on current_agent
            cur = state.get("current_agent")
            if not cur:
                return END
            for i, p in enumerate(self.panel):
                if p.get("name") == cur:
                    return f"sme_{i}"
            return "sme_0"

        builder.add_conditional_edges(
            "route",
            from_route_cond,
            {"done": "done", "sme_0": "sme_0", "sme_1": "sme_1", "sme_2": "sme_2", "__end__": END},
        )

        # From SME nodes, if they set mode to route (after feedback), go back to route; else end (after ask)
        def from_sme_cond(state: InterviewState):
            if state.get("mode") == "route":
                return "route"
            return END

        for name, _node, _ in sme_nodes:
            builder.add_conditional_edges(name, from_sme_cond, {"route": "route", "__end__": END})

        # done -> END
        builder.add_edge("done", END)

        self._graph = builder
        self._app = builder.compile()

        # Initialize state
        self.state = {
            "prompts": self.panel,
            "resume_notes": self.resume_notes,
            "history": [],
            "turn_index": 0,
            "max_turns": self.max_turns,
            "mode": "route",
            "force_stop": False,
        }

    def next_question(self) -> Tuple[str, str, bool]:
        """Run routing + SME ask; returns (question, agent_name, done)."""
        if self._app is None:
            raise RuntimeError("Graph not initialized. Call init_graph first.")
        # Ensure we're ready to ask
        self.state["mode"] = "route"
        self.state = self._app.invoke(self.state)
        # Now in SME ask, invoke again to produce question and halt at await_answer
        self.state = self._app.invoke(self.state)
        agent = self.state.get("current_agent", "Expert")
        question = self.state.get("last_question", "")
        done = self.state.get("mode") == "done"
        return question, agent, done

    def process_user_answer(self, user_answer: str) -> bool:
        if self._app is None:
            raise RuntimeError("Graph not initialized. Call init_graph first.")
        # Provide answer and request feedback
        self.state["user_answer"] = user_answer
        self.state["mode"] = "feedback"
        self.state = self._app.invoke(self.state)
        # One turn completed
        self.state["turn_index"] = self.state.get("turn_index", 0) + 1
        # Route again to see if we should end
        self.state = self._app.invoke(self.state)
        return self.state.get("mode") == "done"

    def stop(self):
        self.state["force_stop"] = True

    def summary(self) -> str:
        # Build final summary with LLM using accumulated history
        if not self.state.get("history"):
            return "No interview conducted."
        items = []
        for h in self.state.get("history", []):
            items.append(f"Q: {h.get('question','')}\nA: {h.get('user_answer','')}\nFeedback: {h.get('feedback','')}")
        context = "\n\n".join(items)
        prompt = (
            "Using the interview transcript and feedback, write a concise summary for the candidate: "
            "What went well and what to improve. Include actionable tips."
        )
        return (self.llm.invoke("You are an interview coach who writes concise, helpful summaries.", context + "\n\n" + prompt) or "").strip()

    # --- Streaming question generation ---
    def _compose_context(self) -> str:
        items = []
        if self.state.get("resume_notes"):
            items.append("Resume Notes:\n" + self.state["resume_notes"])
        recent = self.state.get("history", [])[-5:]
        if recent:
            items.append("History (most recent last):\n" + "\n".join([
                f"Q by {h.get('agent')}: {h.get('question')}\nUser: {h.get('user_answer','')}" for h in recent
            ]))
        return "\n\n".join(items)

    def stream_next_question(self, socketio, room: str) -> Tuple[str, str, bool]:
        """Route to next agent and stream the generated question via Socket.IO.
        Emits: 'question_start', multiple 'question_chunk', then 'question_end'.
        Returns (full_question, agent_name, done).
        """
        # Route to select agent
        self.state["mode"] = "route"
        self.state = self._app.invoke(self.state)
        if self.state.get("mode") == "done":
            return "", "", True
        agent = self.state.get("current_agent", "Expert")
        # Emit start
        socketio.emit('question_start', {"agent": agent}, to=room)

        # Prepare streaming request
        idx = 0
        for i, p in enumerate(self.panel):
            if p.get("name") == agent:
                idx = i
                break
        system_prompt = self.panel[idx].get("system_prompt", "You are an expert interviewer.")
        context = self._compose_context()
        ask_instr = (
            "Craft the next interview question. Ask a single, clear, challenging question tailored to the role and context."
        )

        full_text: List[str] = []
        for piece in self.llm.invoke(system_prompt, context + "\n\n" + ask_instr, stream=True):
            full_text.append(piece)
            socketio.emit('question_chunk', {"text": piece}, to=room)

        question = ("".join(full_text)).strip()
        self.state["last_question"] = question
        self.state["mode"] = "await_answer"
        socketio.emit('question_end', {"agent": agent}, to=room)
        done = self.state.get("mode") == "done"
        return question, agent, done
