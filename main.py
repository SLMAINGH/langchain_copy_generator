import os
import uuid
import json
import threading
import requests
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

# --- IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

app = Flask(__name__)
jobs = {}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY") 

# --- LOGGING ---
def log_to_job(job_id, source, message, type="info"):
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if type == "decision":
        # Supervisor
        html = (
            f"<div class='mt-6 mb-2 p-3 bg-indigo-950 border border-indigo-500 rounded-lg shadow-lg'>"
            f"<div class='flex justify-between items-center mb-2'>"
            f"  <span class='text-indigo-300 font-bold text-xs tracking-widest'>🛡️ SUPERVISOR</span>"
            f"  <span class='text-gray-500 text-[10px]'>{timestamp}</span>"
            f"</div>"
            f"<div class='text-white font-mono text-sm font-semibold'>{message}</div>"
            f"</div>"
        )
    elif type == "critique":
        # The QA step (Red box)
        html = (
            f"<div class='ml-6 mt-2 mb-2 p-3 bg-red-900/20 border-l-4 border-red-500 font-mono text-xs text-gray-300'>"
            f"<div class='text-red-400 font-bold mb-1'>❌ QUALITY CHECK FAILED:</div>"
            f"{message}"
            f"</div>"
        )
    elif type == "success":
        # QA Passed (Green box)
        html = (
            f"<div class='ml-6 mt-2 mb-2 p-2 bg-green-900/20 border-l-4 border-green-500 font-mono text-xs text-gray-300'>"
            f"<span class='text-green-400 font-bold'>✅ QUALITY CHECK PASSED</span>"
            f"</div>"
        )
    elif type == "thought":
        # Worker Output
        html = (
            f"<div class='ml-6 mt-2 mb-2 p-3 bg-gray-800/50 border-l-2 border-gray-600 font-mono text-xs text-gray-300'>"
            f"<div class='text-blue-400 font-bold mb-1'>{source} OUTPUT:</div>"
            f"{message}"
            f"</div>"
        )
    else:
        # System
        html = (
            f"<div class='mt-1 mb-1 text-xs text-gray-400'>"
            f"<span class='font-bold opacity-70'>[{timestamp}] {source}:</span> {message}"
            f"</div>"
        )
    
    if job_id in jobs:
        jobs[job_id]['logs'].append(html)

# --- HELPER: VERIFIER (The Critic) ---
def verify_output(llm, job_id, task_name, content, criteria):
    """
    Returns (bool_passed, string_critique)
    """
    log_to_job(job_id, "QA_BOT", f"Auditing {task_name}...", type="info")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Strict Quality Auditor.
        
        TASK: {task_name}
        CONTENT SUBMITTED:
        {content}
        
        QUALITY CRITERIA:
        {criteria}
        
        1. Evaluate if the content meets the criteria.
        2. If NO, provide specific instructions on how to fix it.
        3. If YES, return "PASS".
        
        Return JSON:
        {{
            "status": "PASS" or "FAIL",
            "critique": "Short explanation of what is missing (or 'Looks good' if passed)"
        }}
        """
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        res = chain.invoke({"task_name": task_name, "content": str(content)[:3000], "criteria": criteria})
        return (res['status'] == "PASS", res['critique'])
    except:
        return (True, "Auto-passed due to format error")

# --- WORKERS (Now accept 'feedback') ---

def worker_research(llm, job_id, state, feedback=None):
    company = state.get("company_name")
    log_msg = f"Researching {company}..."
    if feedback:
        log_msg += f" (Attempting to fix: {feedback})"
    
    log_to_job(job_id, "RESEARCHER", log_msg, type="info")

    # If feedback exists, we inject it into the prompt to guide the LLM
    additional_instruction = ""
    if feedback:
        additional_instruction = f"IMPORTANT - PREVIOUS ATTEMPT REJECTED. FIX THIS: {feedback}"

    # (Mocking the Perplexity call for brevity, but you'd inject 'additional_instruction' into messages)
    # In real use: add logic to query differently if feedback is present
    if not PERPLEXITY_API_KEY:
        return f"Simulated Research for {company}. {additional_instruction}"

    # ... Your Perplexity Code Here ...
    # For demo purposes, we return a string
    return f"Detailed financial analysis of {company}. Focus on AI adoption. {additional_instruction}"

def worker_writer(llm, job_id, state, feedback=None):
    log_to_job(job_id, "WRITER", "Writing draft...", type="info")
    
    prompt_text = """Write a LinkedIn message."""
    if feedback:
        prompt_text += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS DRAFT: {feedback}\nPlease fix this."
    
    # Simple chain
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({})
    
    log_to_job(job_id, "WRITER", result, type="thought")
    return result

# --- MAIN WORKFLOW LOOP ---
def process_workflow(job_id, input_data):
    job = jobs[job_id]
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.0)

    # STATE
    state = {
        "company_name": input_data['input_json'].get('records', [])[0].get('companyName', "Target Co"),
        "research": None,
        "copy": None,
        
        # New: Tracking Retries
        "retry_counts": {"RESEARCHER": 0, "WRITER": 0},
        "current_feedback": None # Stores the critique to pass to the worker
    }
    
    # DEFINE PHASES & CRITERIA
    # We linearize the phases for the "Supervisor" to manage easily
    phases = [
        {
            "role": "RESEARCHER", 
            "key": "research", 
            "criteria": "Must include specific 'Financial Challenges' and 'Tech Stack'. Must be over 100 words." 
        },
        {
            "role": "WRITER", 
            "key": "copy", 
            "criteria": "Must be under 300 characters. Must NOT sound robotic. Must mention the financial challenges found."
        }
    ]
    
    current_phase_idx = 0
    MAX_RETRIES = 3

    try:
        while current_phase_idx < len(phases):
            phase = phases[current_phase_idx]
            role = phase['role']
            key = phase['key']
            
            # 1. SUPERVISOR DECISION
            if state[key] is None:
                # Task hasn't been done yet, OR it was wiped due to retry
                log_to_job(job_id, "SUPERVISOR", f"Assigning task to {role}.", type="decision")
                
                if role == "RESEARCHER":
                    out = worker_research(llm, job_id, state, state['current_feedback'])
                elif role == "WRITER":
                    out = worker_writer(llm, job_id, state, state['current_feedback'])
                
                state[key] = out # Save output
                state['current_feedback'] = None # Clear feedback after usage
            
            else:
                # 2. VERIFICATION STEP
                # The output exists, let's verify it
                passed, critique = verify_output(llm, job_id, role, state[key], phase['criteria'])
                
                if passed:
                    log_to_job(job_id, "QA", "Approved.", type="success")
                    current_phase_idx += 1 # Move to next phase
                    state['current_feedback'] = None
                else:
                    # FAILED
                    retries = state['retry_counts'][role]
                    if retries < MAX_RETRIES:
                        state['retry_counts'][role] += 1
                        state['current_feedback'] = critique
                        state[key] = None # WIPE the bad output so loop runs worker again
                        
                        log_to_job(job_id, "QA", f"{critique} (Retry {retries+1}/{MAX_RETRIES})", type="critique")
                        log_to_job(job_id, "SUPERVISOR", f"Re-assigning {role} to fix errors.", type="decision")
                    else:
                        log_to_job(job_id, "QA", f"Max retries reached. Proceeding with suboptimal result.", type="info")
                        current_phase_idx += 1

            # Progress update
            job['progress'] = int(((current_phase_idx) / len(phases)) * 100)

        # FINISH
        job['result'] = state['copy']
        job['status'] = "completed"
        job['progress'] = 100
        log_to_job(job_id, "SYSTEM", "Workflow Complete.")

    except Exception as e:
        job['status'] = "failed"
        log_to_job(job_id, "SYSTEM", f"Critical Error: {e}")
        print(e)

# --- FLASK SETUP (Keep your existing Routes & HTML Template) ---
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/start', methods=['POST'])
def start_job():
    data = request.json
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "status": "running", "logs": [], "result": None, "progress": 0}
    threading.Thread(target=process_workflow, args=(job_id, data)).start()
    return jsonify({"job_id": job_id})

@app.route('/api/status/<job_id>')
def status(job_id):
    return jsonify(jobs.get(job_id, {"error": "not found"}))

# NOTE: Use the HTML_TEMPLATE from the previous response.
# It works perfectly with these log types.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Supervisor Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .log-container { font-family: 'Menlo', monospace; font-size: 0.8em; }
        .blink { animation: blinker 1s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
        /* Dots animation */
        .loading-dots:after { content: ' .'; animation: dots 1s steps(5, end) infinite; }
        @keyframes dots { 0%, 20% { content: ' .'; } 40% { content: ' ..'; } 60% { content: ' ...'; } 80%, 100% { content: ''; } }
    </style>
</head>
<body class="bg-gray-900 text-white h-screen flex flex-col overflow-hidden">
    <div class="flex-1 flex flex-col max-w-7xl mx-auto w-full p-6 gap-6">
        <div class="flex justify-between items-center">
            <h1 class="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400">
                🤖 Supervisor Architecture (Self-Correcting)
            </h1>
            <div id="statusBadge" class="bg-gray-800 px-4 py-1 rounded-full text-xs font-mono uppercase tracking-widest text-gray-500">Idle</div>
        </div>
        <div class="grid grid-cols-12 gap-6 flex-1 min-h-0">
            <div class="col-span-3 bg-gray-800 rounded-xl p-4 flex flex-col gap-4 border border-gray-700">
                <label class="text-xs font-bold text-gray-400 uppercase">Input Context</label>
                <textarea id="jsonInput" class="flex-1 bg-gray-900 rounded p-3 text-xs font-mono text-green-400 outline-none resize-none border border-gray-700 focus:border-indigo-500 transition" placeholder='{"records": [{"companyName": "NVIDIA"}]}'></textarea>
                <button onclick="startJob()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 rounded transition shadow-lg shadow-indigo-500/20">Start Workflow</button>
            </div>
            <div class="col-span-6 bg-[#0B0F19] rounded-xl border border-gray-700 flex flex-col relative overflow-hidden shadow-2xl">
                <div class="bg-gray-800/50 px-4 py-2 border-b border-gray-700 flex justify-between items-center backdrop-blur">
                    <span class="text-xs font-bold text-indigo-400 uppercase tracking-widest">Live Execution Trace</span>
                    <span id="pct" class="text-xs font-mono text-gray-500">0%</span>
                </div>
                <div id="logs" class="log-container flex-1 p-4 overflow-y-auto pb-12 space-y-2"></div>
                <div id="thinking" class="hidden absolute bottom-4 left-6 text-indigo-400 text-xs font-mono bg-gray-900/90 px-3 py-1 rounded border border-indigo-500/30">
                    <span class="loading-dots">Supervisor is auditing</span>
                </div>
            </div>
            <div class="col-span-3 bg-white text-gray-900 rounded-xl p-6 flex flex-col shadow-xl overflow-hidden">
                <h3 class="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">Final Deliverable</h3>
                <div id="result" class="flex-1 overflow-y-auto text-sm whitespace-pre-wrap font-sans leading-relaxed"></div>
            </div>
        </div>
    </div>
    <script>
        let jobId = null, timer = null;
        async function startJob() {
            const input = document.getElementById('jsonInput').value;
            try { JSON.parse(input); } catch { return alert("Invalid JSON"); }
            const res = await fetch('/api/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({input_json: JSON.parse(input)}) });
            const data = await res.json();
            jobId = data.job_id;
            document.getElementById('logs').innerHTML = '';
            document.getElementById('statusBadge').innerText = "RUNNING";
            document.getElementById('statusBadge').classList.add('text-indigo-400', 'blink');
            document.getElementById('thinking').classList.remove('hidden');
            timer = setInterval(poll, 1000);
        }
        async function poll() {
            if (!jobId) return;
            const res = await fetch(`/api/status/${jobId}`);
            const data = await res.json();
            const logs = document.getElementById('logs');
            logs.innerHTML = data.logs.join('');
            logs.scrollTop = logs.scrollHeight;
            document.getElementById('pct').innerText = data.progress + "%";
            if (data.status !== 'running') {
                clearInterval(timer);
                document.getElementById('statusBadge').innerText = data.status.toUpperCase();
                document.getElementById('statusBadge').classList.remove('blink');
                document.getElementById('thinking').classList.add('hidden');
                if(data.result) document.getElementById('result').innerText = data.result;
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
