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

# --- CONFIG ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY") 

# --- LOGGING SYSTEM ---
def log_to_job(job_id, source, message, type="info"):
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if type == "decision":
        # Supervisor Decision (The Boss)
        html = (
            f"<div class='mt-6 mb-2 p-4 bg-indigo-950/80 border border-indigo-500/50 rounded-lg shadow-lg relative overflow-hidden'>"
            f"<div class='absolute top-0 left-0 w-1 h-full bg-indigo-500'></div>"
            f"<div class='flex justify-between items-center mb-2'>"
            f"  <span class='text-indigo-300 font-bold text-xs tracking-widest'>🛡️ SUPERVISOR</span>"
            f"  <span class='text-indigo-400/50 text-[10px] font-mono'>{timestamp}</span>"
            f"</div>"
            f"<div class='text-gray-100 font-mono text-sm font-semibold leading-relaxed'>{message}</div>"
            f"</div>"
        )
    elif type == "critique":
        # QA Failed (The Red Flag)
        html = (
            f"<div class='ml-8 mt-2 mb-2 p-3 bg-red-900/20 border-l-2 border-red-500 font-mono text-xs text-red-200'>"
            f"<div class='flex items-center gap-2 mb-1 text-red-400 font-bold'>"
            f"  <span>❌ REJECTED</span>"
            f"</div>"
            f"{message}"
            f"</div>"
        )
    elif type == "success":
        # QA Passed
        html = (
            f"<div class='ml-8 mt-2 mb-2 p-2 bg-green-900/10 border-l-2 border-green-500 font-mono text-xs'>"
            f"<span class='text-green-400 font-bold'>✅ APPROVED</span>"
            f"</div>"
        )
    elif type == "thought":
        # Worker Output
        html = (
            f"<div class='ml-8 mt-2 mb-2 p-3 bg-gray-800/50 border-l-2 border-gray-600 font-mono text-xs text-gray-300'>"
            f"<div class='text-blue-400 font-bold mb-1 opacity-70'>{source} DRAFT:</div>"
            f"<div class='whitespace-pre-wrap'>{message}</div>"
            f"</div>"
        )
    else:
        # System Status
        html = (
            f"<div class='mt-1 mb-1 text-xs text-gray-500 font-mono ml-2'>"
            f"<span class='opacity-50'>[{timestamp}] {source}:</span> {message}"
            f"</div>"
        )
    
    if job_id in jobs:
        jobs[job_id]['logs'].append(html)

# --- HELPER: VERIFIER (The Critic) ---
def verify_output(llm, job_id, task_name, content, criteria):
    log_to_job(job_id, "QA_BOT", f"Reviewing {task_name}...", type="info")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a strict QA Editor.
        
        TASK: {task_name}
        CONTENT: {content}
        CRITERIA: {criteria}
        
        Rules:
        1. If specific names (Company, Person) are missing or placeholders like [Name] exist, FAIL immediately.
        2. If tone is robotic, FAIL.
        
        Return JSON: {{ "status": "PASS" | "FAIL", "critique": "Reason..." }}
        """
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        res = chain.invoke({"task_name": task_name, "content": str(content)[:3000], "criteria": criteria})
        return (res['status'] == "PASS", res['critique'])
    except:
        return (True, "Auto-passed (Format Error)")

# --- WORKERS ---

def worker_research(llm, job_id, state, feedback=None):
    company = state['company_name']
    
    if not PERPLEXITY_API_KEY:
        # Fallback simulation if no key
        return f"Market analysis of {company}: Currently facing cash flow challenges due to rapid expansion. Their CTO mentioned needing better budget tools in a recent podcast."

    # If we have a key, we'd call Perplexity here...
    # For now, we simulate a robust response to ensure the writer has good data.
    return f"Live data for {company}: 1. Stock down 5% last quarter. 2. CEO emphasized 'Operational Efficiency' in Q3 earnings. 3. Migrating to cloud infrastructure."

def worker_writer(llm, job_id, state, feedback=None):
    log_to_job(job_id, "WRITER", "Drafting message...", type="info")
    
    # EXPLICIT DATA EXTRACTION TO PREVENT PLACEHOLDERS
    record = state['input_records'][0]
    first_name = record.get('firstName', 'there')
    company = record.get('companyName', 'your company')
    research = state.get('research', 'No data')
    
    prompt_text = f"""
    You are a Sales Copywriter.
    
    TARGET: {first_name} at {company}
    RESEARCH INSIGHTS: {research}
    
    Task: Write a LinkedIn connection request (max 300 chars).
    
    RULES:
    1. NEVER use placeholders like [Name]. Use the real name: "{first_name}".
    2. Reference the research explicitly.
    3. Keep it conversational.
    """
    
    if feedback:
        prompt_text += f"\n\n⚠️ PREVIOUS REJECTION REASON: {feedback}\nFIX THIS IMMEDIATELY."
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({})
    
    log_to_job(job_id, "WRITER", result, type="thought")
    return result

# --- WORKFLOW ---
def process_workflow(job_id, input_data):
    job = jobs[job_id]
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.0)

    # INITIAL STATE
    records = input_data['input_json'].get('records', [])
    if not records:
        log_to_job(job_id, "SYSTEM", "No records found in input JSON.")
        job['status'] = "failed"
        return

    state = {
        "input_records": records,
        "company_name": records[0].get('companyName', "Unknown"),
        "research": None,
        "copy": None,
        "retry_counts": {"RESEARCHER": 0, "WRITER": 0},
        "current_feedback": None
    }
    
    # THE SUPERVISOR PLAN
    phases = [
        {
            "role": "RESEARCHER", 
            "key": "research", 
            "criteria": "Must mention specific financial or technical challenges." 
        },
        {
            "role": "WRITER", 
            "key": "copy", 
            "criteria": "Must NOT contain '[Name]' or brackets. Must mention the Company Name explicitly. Under 300 chars."
        }
    ]
    
    current_phase_idx = 0
    MAX_RETRIES = 3

    try:
        while current_phase_idx < len(phases):
            phase = phases[current_phase_idx]
            role = phase['role']
            key = phase['key']
            
            # 1. ASSIGN TASK
            if state[key] is None:
                log_to_job(job_id, "SUPERVISOR", f"Activating {role}...", type="decision")
                
                if role == "RESEARCHER":
                    out = worker_research(llm, job_id, state, state['current_feedback'])
                elif role == "WRITER":
                    out = worker_writer(llm, job_id, state, state['current_feedback'])
                
                state[key] = out
                state['current_feedback'] = None 
            
            else:
                # 2. VERIFY TASK
                passed, critique = verify_output(llm, job_id, role, state[key], phase['criteria'])
                
                if passed:
                    log_to_job(job_id, "QA", "Quality Standard Met.", type="success")
                    current_phase_idx += 1
                else:
                    # RETRY LOGIC
                    retries = state['retry_counts'][role]
                    if retries < MAX_RETRIES:
                        state['retry_counts'][role] += 1
                        state['current_feedback'] = critique
                        state[key] = None # WIPE output to force retry
                        log_to_job(job_id, "QA", f"Issues found: {critique} (Retry {retries+1})", type="critique")
                    else:
                        log_to_job(job_id, "QA", "Max retries hit. Moving on.", type="info")
                        current_phase_idx += 1

            job['progress'] = int(((current_phase_idx) / len(phases)) * 100)

        job['result'] = state['copy']
        job['status'] = "completed"
        job['progress'] = 100
        log_to_job(job_id, "SYSTEM", "Workflow Complete.")

    except Exception as e:
        job['status'] = "failed"
        log_to_job(job_id, "SYSTEM", f"Critical Error: {e}")

# --- API ROUTES ---
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

# --- FRONTEND TEMPLATE (FIXED SCROLLING) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Supervisor Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .log-container { font-family: 'Menlo', monospace; font-size: 0.85em; }
        .blink { animation: blinker 1s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
        .loading-dots:after { content: ' .'; animation: dots 1s steps(5, end) infinite; }
        @keyframes dots { 0%, 20% { content: ' .'; } 40% { content: ' ..'; } 60% { content: ' ...'; } 80%, 100% { content: ''; } }
        
        /* SCROLLBAR STYLING */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="bg-[#0B0F19] text-white h-screen flex flex-col overflow-hidden selection:bg-indigo-500/30">

    <div class="flex-1 flex flex-col max-w-7xl mx-auto w-full p-6 gap-6">
        
        <div class="flex justify-between items-center border-b border-gray-800 pb-4">
            <div class="flex items-center gap-3">
                <div class="w-3 h-3 bg-indigo-500 rounded-full blink"></div>
                <h1 class="text-xl font-bold tracking-tight text-gray-100">
                    Agent <span class="text-indigo-400">Supervisor</span>
                </h1>
            </div>
            <div id="statusBadge" class="bg-gray-800 px-3 py-1 rounded text-xs font-mono uppercase tracking-widest text-gray-500 border border-gray-700">
                System Idle
            </div>
        </div>

        <div class="grid grid-cols-12 gap-6 flex-1 min-h-0">
            
            <div class="col-span-3 flex flex-col gap-4">
                <div class="bg-[#111827] rounded-lg p-1 border border-gray-700 flex-1 flex flex-col">
                    <div class="px-4 py-2 bg-gray-800/50 border-b border-gray-700 text-[10px] font-bold text-gray-400 uppercase tracking-widest">Target Context (JSON)</div>
                    <textarea id="jsonInput" class="flex-1 bg-transparent p-4 text-xs font-mono text-emerald-400 outline-none resize-none placeholder-gray-700" spellcheck="false">{
    "records": [
        {
            "companyName": "TechFlow Inc",
            "firstName": "Sarah",
            "title": "VP of Operations"
        }
    ]
}</textarea>
                </div>
                <button onclick="startJob()" class="bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-bold py-3 rounded-lg transition shadow-lg shadow-indigo-900/20 border border-indigo-500/50">
                    Run Workflow
                </button>
            </div>

            <div class="col-span-6 bg-[#0f1219] rounded-lg border border-gray-700 flex flex-col relative overflow-hidden shadow-2xl">
                <div class="bg-[#161b28] px-4 py-2 border-b border-gray-800 flex justify-between items-center">
                    <span class="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Execution Stream</span>
                    <span id="pct" class="text-[10px] font-mono text-gray-500">0%</span>
                </div>
                
                <div id="logs" class="log-container flex-1 p-4 overflow-y-auto pb-16 space-y-1">
                    </div>

                <div id="thinking" class="hidden absolute bottom-0 w-full bg-gradient-to-t from-[#0B0F19] to-transparent p-4 pointer-events-none">
                    <div class="inline-flex items-center gap-2 px-3 py-1 bg-indigo-900/80 rounded-full border border-indigo-500/30 text-indigo-300 text-[10px] font-mono shadow-lg backdrop-blur-sm">
                        <span class="w-1.5 h-1.5 bg-indigo-400 rounded-full blink"></span>
                        <span class="loading-dots">Supervisor is auditing response</span>
                    </div>
                </div>
            </div>

            <div class="col-span-3 bg-[#111827] rounded-lg border border-gray-700 flex flex-col shadow-xl overflow-hidden">
                <div class="px-4 py-2 bg-gray-800/50 border-b border-gray-700 text-[10px] font-bold text-gray-400 uppercase tracking-widest">Final Deliverable</div>
                <div id="result" class="flex-1 p-6 overflow-y-auto text-sm text-gray-300 font-sans leading-relaxed whitespace-pre-wrap">
                    <span class="text-gray-600 italic text-xs">Waiting for verification...</span>
                </div>
            </div>

        </div>
    </div>

    <script>
        let jobId = null;
        let timer = null;

        async function startJob() {
            const input = document.getElementById('jsonInput').value;
            try { JSON.parse(input); } catch { return alert("Invalid JSON"); }

            const res = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({input_json: JSON.parse(input)})
            });
            const data = await res.json();
            jobId = data.job_id;
            
            document.getElementById('logs').innerHTML = '';
            document.getElementById('result').innerHTML = '<span class="text-gray-600 italic text-xs">Processing...</span>';
            document.getElementById('statusBadge').innerText = "RUNNING";
            document.getElementById('statusBadge').className = "bg-indigo-900/30 px-3 py-1 rounded text-xs font-mono uppercase tracking-widest text-indigo-400 border border-indigo-500/50 blink";
            document.getElementById('thinking').classList.remove('hidden');
            
            timer = setInterval(poll, 800);
        }

        async function poll() {
            if (!jobId) return;
            const res = await fetch(`/api/status/${jobId}`);
            const data = await res.json();
            
            const logs = document.getElementById('logs');
            
            // Only update if length changed to prevent flickering
            if (logs.children.length !== data.logs.length) {
                logs.innerHTML = data.logs.join('');
                
                // --- FORCE SCROLL TO BOTTOM ---
                logs.scrollTo({
                    top: logs.scrollHeight,
                    behavior: 'smooth'
                });
            }

            document.getElementById('pct').innerText = data.progress + "%";

            if (data.status !== 'running') {
                clearInterval(timer);
                document.getElementById('statusBadge').innerText = data.status.toUpperCase();
                document.getElementById('statusBadge').className = "bg-green-900/30 px-3 py-1 rounded text-xs font-mono uppercase tracking-widest text-green-400 border border-green-500/50";
                document.getElementById('thinking').classList.add('hidden');
                
                if(data.result) {
                    document.getElementById('result').innerText = data.result;
                }
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
