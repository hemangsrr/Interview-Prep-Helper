import os
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from dotenv import load_dotenv
from services.pdf_utils import extract_text_from_pdf
from agents.panel_builder import propose_panel_from_jd
from agents.langgraph_orchestrator import LangGraphOrchestrator
from flask_socketio import SocketIO, join_room
from services.db import (
    save_panel as db_save_panel,
    save_interview_state as db_save_state,
    save_jd_panel_with_embedding,
    find_similar_jd_panel,
    get_interview_state,
    get_panel as db_get_panel,
)
from services.pdf_report import build_feedback_pdf
from markdown import markdown as md_to_html
from llm import LLM

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit
ALLOWED_PDF_MIMES = {"application/pdf"}
socketio = SocketIO(app, cors_allowed_origins="*")

# Simple in-memory store for interview state per session
INTERVIEWS = {}


def _get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid4())
    return session["sid"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze_jd", methods=["POST"]) 
def analyze_jd():
    jd_text = request.form.get("jd_text", "").strip()
    file = request.files.get("jd_pdf")
    if file and file.filename:
        # Guard: only PDFs
        if file.mimetype not in ALLOWED_PDF_MIMES or not file.filename.lower().endswith('.pdf'):
            flash("Only PDF files are allowed for JD upload.", "error")
            return redirect(url_for("index"))
        try:
            jd_text = extract_text_from_pdf(file)
        except Exception as e:
            flash(f"Failed to read PDF: {e}", "error")
            return redirect(url_for("index"))
    if not jd_text:
        flash("Please provide a Job Description as text or PDF.", "error")
        return redirect(url_for("index"))

    llm = LLM()
    # Semantic reuse: check for similar JD panels in DB using embeddings
    try:
        jd_emb = llm.embed(jd_text)
        match = find_similar_jd_panel(jd_emb, threshold=0.9)
    except Exception:
        jd_emb = None
        match = None

    if match:
        panel = match.get("panel", [])
        flash("Similar JD found in DB. Reusing its panel to save tokens.", "success")
    else:
        panel = propose_panel_from_jd(llm, jd_text)
    session["panel"] = panel  # list of {name, system_prompt}
    session["jd_text"] = jd_text
    # persist proposed/reused panel
    sid = _get_session_id()
    try:
        if jd_emb is not None:
            save_jd_panel_with_embedding(sid, jd_text, panel, jd_emb)
        else:
            db_save_panel(sid, panel, jd_text)
    except Exception:
        pass
    return redirect(url_for("panel"))


@app.route("/panel")
def panel():
    panel = session.get("panel")
    if not panel:
        return redirect(url_for("index"))
    return render_template("panel.html", panel=panel)


@app.route("/save_panel", methods=["POST"]) 
def save_panel():
    panel = []
    for i in range(3):
        name = request.form.get(f"name_{i}", f"Expert {i+1}")
        prompt = request.form.get(f"prompt_{i}", "").strip()
        panel.append({"name": name, "system_prompt": prompt})
    session["panel"] = panel
    # update DB stored panel for this sid
    try:
        sid = _get_session_id()
        db_save_panel(sid, panel, session.get("jd_text", ""))
    except Exception:
        pass
    return redirect(url_for("interview_setup"))


@app.route("/interview_setup")
def interview_setup():
    if not session.get("panel"):
        return redirect(url_for("index"))
    return render_template("interview_setup.html")


@app.route("/upload_resume", methods=["POST"]) 
def upload_resume():
    if not session.get("panel"):
        return redirect(url_for("index"))
    resume_text = request.form.get("resume_text", "").strip()
    file = request.files.get("resume_pdf")
    if file and file.filename:
        if file.mimetype not in ALLOWED_PDF_MIMES or not file.filename.lower().endswith('.pdf'):
            flash("Only PDF files are allowed for resume upload.", "error")
            return redirect(url_for("interview_setup"))
        try:
            resume_text = extract_text_from_pdf(file)
        except Exception as e:
            flash(f"Failed to read resume PDF: {e}", "error")
            return redirect(url_for("interview_setup"))

    notes = None
    if resume_text:
        llm = LLM()
        # Summarize key points for agents to use
        prompt = (
            "Extract concise key points from this resume that are useful for interviewers. "
            "Return 5-10 bullet points.\n\nResume:\n" + resume_text[:15000]
        )
        notes = llm.invoke("You extract key points as short bullet points.", prompt)
    session["resume_notes"] = notes or ""
    flash("Resume analyzed and notes prepared.", "success")
    return redirect(url_for("interview_setup"))


@app.route("/start_interview", methods=["POST"]) 
def start_interview():
    sid = _get_session_id()
    panel = session.get("panel")
    if not panel:
        return redirect(url_for("index"))
    notes = session.get("resume_notes", "")

    # Initialize LangGraph orchestrator
    llm = LLM()
    orchestrator = LangGraphOrchestrator(llm=llm, max_turns=8)
    orchestrator.init_graph(panel=panel, resume_notes=notes)
    INTERVIEWS[sid] = orchestrator  # store instance in memory
    # persist initial state
    try:
        db_save_state(sid, orchestrator.state)
    except Exception:
        pass
    return redirect(url_for("interview"))


@app.route("/interview")
def interview():
    sid = _get_session_id()
    if sid not in INTERVIEWS:
        # Attempt to rehydrate from DB if page reloaded mid-interview
        try:
            state = get_interview_state(sid)
            if state:
                llm = LLM()
                orch = LangGraphOrchestrator(llm=llm, max_turns=state.get("max_turns", 8))
                # Rebuild prompts/resume_notes from state itself (it contains prompts and notes)
                orch.panel = state.get("prompts", [])
                orch.resume_notes = state.get("resume_notes", "")
                orch.init_graph(panel=orch.panel, resume_notes=orch.resume_notes)
                orch.state = state
                INTERVIEWS[sid] = orch
        except Exception:
            pass
    if sid not in INTERVIEWS:
        return redirect(url_for("index"))
    return render_template("interview.html", sid=sid)


@app.route("/api/next_question", methods=["POST"]) 
def api_next_question():
    sid = _get_session_id()
    if sid not in INTERVIEWS:
        return jsonify({"error": "Interview not started"}), 400
    orchestrator = INTERVIEWS[sid]
    question, agent_name, done = orchestrator.next_question()
    try:
        db_save_state(sid, orchestrator.state)
    except Exception:
        pass
    return jsonify({"question": question, "agent": agent_name, "done": done})


@app.route("/api/submit_answer", methods=["POST"]) 
def api_submit_answer():
    sid = _get_session_id()
    if sid not in INTERVIEWS:
        return jsonify({"error": "Interview not started"}), 400
    user_answer = request.json.get("answer", "")
    orchestrator = INTERVIEWS[sid]
    done = orchestrator.process_user_answer(user_answer)
    try:
        db_save_state(sid, orchestrator.state)
    except Exception:
        pass
    return jsonify({"done": done})


@app.route("/api/stop", methods=["POST"]) 
def api_stop():
    sid = _get_session_id()
    orchestrator = INTERVIEWS.get(sid)
    if not orchestrator:
        return jsonify({"ok": True})
    orchestrator.stop()
    return jsonify({"ok": True})


@app.route("/api/summary", methods=["GET"]) 
def api_summary():
    sid = _get_session_id()
    orchestrator = INTERVIEWS.get(sid)
    if not orchestrator:
        return jsonify({"summary": ""})
    summary = orchestrator.summary()
    return jsonify({"summary": summary})


@app.route("/feedback")
def feedback_view():
    sid = _get_session_id()
    orchestrator = INTERVIEWS.get(sid)
    if not orchestrator:
        flash("No interview session found.", "error")
        return redirect(url_for("index"))
    summary = orchestrator.summary()
    summary_html = md_to_html(summary or "", extensions=["extra", "sane_lists"])  # render markdown to HTML
    return render_template("feedback.html", summary_html=summary_html)


@app.route("/download_feedback")
def download_feedback():
    sid = _get_session_id()
    orchestrator = INTERVIEWS.get(sid)
    if not orchestrator:
        flash("No interview session found.", "error")
        return redirect(url_for("index"))
    summary = orchestrator.summary()
    pdf_bytes = build_feedback_pdf(summary)
    from flask import send_file
    from io import BytesIO
    return send_file(BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name='interview_feedback.pdf')


# --- Socket.IO integration ---
@socketio.on('join')
def on_join(data):
    room = data.get('room')
    if room:
        join_room(room)


@app.route("/api/next_question_stream", methods=["POST"]) 
def api_next_question_stream():
    sid = _get_session_id()
    if sid not in INTERVIEWS:
        return jsonify({"error": "Interview not started"}), 400
    orchestrator = INTERVIEWS[sid]

    def _task():
        orchestrator.stream_next_question(socketio, room=sid)
        try:
            db_save_state(sid, orchestrator.state)
        except Exception:
            pass

    socketio.start_background_task(_task)
    return jsonify({"started": True})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
