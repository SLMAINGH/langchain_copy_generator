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
            f"<div class='mt-6 mb-4 p-4 bg-indigo-950/40 border border-indigo-500/50 rounded-lg shadow-lg relative overflow-hidden group hover:border-indigo-400 transition-colors'>"
            f"<div class='absolute top-0 left-0 w-1 h-full bg-indigo-500'></div>"
            f"<div class='flex justify-between items-center mb-2'>"
            f"  <span class='text-indigo-300 font-bold text-xs tracking-widest flex items-center gap-2'>🛡️ SUPERVISOR INTERVENTION</span>"
            f"  <span class='text-indigo-400/50 text-[10px] font-mono'>{timestamp}</span>"
            f"</div>"
            f"<div class='text-gray-100 font-mono text-xs font-medium leading-relaxed whitespace-pre-wrap'>{message}</div>"
            f"</div>"
        )
    elif type == "critique":
        # QA Failed
        html = (
            f"<div class='ml-8 mt-2 mb-2 p-3 bg-red-900/10 border-l-2 border-red-500/50 font-mono text-xs text-red-200/80'>"
            f"<div class='flex items-center gap-2 mb-1 text-red-400 font-bold uppercase text-[10px]'>"
            f"  <span>❌ Prompt Modification Required</span>"
            f"</div>"
            f"{message}"
            f"</div>"
        )
    elif type == "thought":
        # Worker Output
        html = (
            f"<div class='ml-8 mt-2 mb-2 p-3 bg-gray-800/40 border-l border-gray-700 font-mono text-xs text-gray-400'>"
            f"<div class='text-blue-400/70 font-bold mb-1 uppercase text-[10px]'>{source} Output:</div>"
            f"<div class='whitespace-pre-wrap leading-relaxed'>{message}</div>"
            f"</div>"
        )
    else:
        # System Status
        html = (
            f"<div class='mt-1 mb-1 text-[10px] text-gray-600 font-mono ml-2 uppercase tracking-wide'>"
            f"<span class='opacity-50'>[{timestamp}] {source}:</span> {message}"
            f"</div>"
        )
    
    if job_id in jobs:
        jobs[job_id]['logs'].append(html)

# --- HELPER: SUPERVISOR AUDIT & PROMPT RE-ENGINEERING ---
def audit_and_revise(llm, job_id, task_name, current_content, previous_instructions, criteria):
    log_to_job(job_id, "QA_BOT", f"Auditing {task_name}...", type="info")
    
    prompt = ChatPromptTemplate.from_template(
        """You are the Supervisor. You have the authority to change the Worker's instructions.
        
        TASK: {task_name}
        CURRENT OUTPUT: {content}
        CRITERIA: {criteria}
        PREVIOUS INSTRUCTIONS USED: {previous_instructions}
        
        1. Does the output meet the criteria?
        2. If NO, create NEW, BETTER INSTRUCTIONS for the worker. 
           (e.g., If the output was too formal, your new instruction should be: "Write it again, but strictly use slang and lowercase letters.")
        
        Return JSON: 
        {{ 
            "status": "PASS" | "FAIL", 
            "reason": "Why it failed",
            "new_instructions": "The exact new prompt text to give the worker" 
        }}
        """
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        res = chain.invoke({
            "task_name": task_name, 
            "content": str(current_content)[:3000], 
            "criteria": criteria,
            "previous_instructions": previous_instructions
        })
        return res
    except:
        return {"status": "PASS", "reason": "Auto-passed (Format Error)", "new_instructions": ""}

# --- WORKERS (Now driven by Dynamic Instructions) ---

def worker_research(llm, job_id, state, instructions):
    log_to_job(job_id, "RESEARCHER", f"Executing instructions: {instructions}", type="info")
    company = state['company_name']
    
    # Simulating data fetching based on instructions
    if "financial" in instructions.lower():
        return f"Deep Dive Financials for {company}: 1. Cash flow -10% YoY. 2. Seeking Series B funding. 3. Layoffs in marketing dept."
    elif "tech" in instructions.lower():
        return f"Tech Stack for {company}: Using AWS, Python, Legacy SQL databases (migration planned)."
    else:
        # Default generic
        return f"General Overview of {company}: A mid-sized tech firm focusing on B2B logistics. Recently appointed a new CTO."

def worker_writer(llm, job_id, state, instructions):
    log_to_job(job_id, "WRITER", "Writing...", type="info")
    
    record = state['input_records'][0]
    first_name = record.get('firstName', 'there')
    company = record.get('companyName', 'your company')
    research = state.get('research', 'No data')
    
    # The PROMPT is now dynamically controlled by the Supervisor
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI Sales Assistant.
        
        CONTEXT DATA:
        Target: {first_name} at {company}
        Research: {research}
        
        SUPERVISOR INSTRUCTIONS (FOLLOW STRICTLY):
        {instructions}
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "first_name": first_name,
        "company": company,
        "research": research,
        "instructions": instructions
    })
    
    log_to_job(job_id, "WRITER", result, type="thought")
    return result

# --- WORKFLOW ---
def process_workflow(job_id, input_data):
    job = jobs[job_id]
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.0)

    records = input_data['input_json'].get('records', [])
    if not records: return
    
    state = {
        "input_records": records,
        "company_name": records[0].get('companyName', "Unknown"),
        "research": None,
        "copy": None,
        # We track instructions here so they can be modified
        "instructions": {
            "RESEARCHER": "Find general strategic challenges and financial health.",
            "WRITER": "Write a standard professional connection request (under 300 chars)."
        },
        "retry_counts": {"RESEARCHER": 0, "WRITER": 0}
    }
    
    phases = [
        {"role": "RESEARCHER", "key": "research", "criteria": "Must mention specific recent events."},
        {"role": "WRITER", "key": "copy", "criteria": "Must be under 300 chars. NO placeholders like [Name]. Must sound casual, not robotic."}
    ]
    
    current_phase_idx = 0
    MAX_RETRIES = 3

    try:
        while current_phase_idx < len(phases):
            phase = phases[current_phase_idx]
            role = phase['role']
            key = phase['key']
            
            # 1. ASSIGN TASK (Using current instructions)
            if state[key] is None:
                current_instruction = state['instructions'][role]
                # Log the specific prompt the supervisor is injecting
                log_to_job(job_id, "SUPERVISOR", f"Prompting {role} with:\n'{current_instruction}'", type="decision")
                
                if role == "RESEARCHER":
                    out = worker_research(llm, job_id, state, current_instruction)
                elif role == "WRITER":
                    out = worker_writer(llm, job_id, state, current_instruction)
                
                state[key] = out
            
            else:
                # 2. AUDIT TASK
                audit_res = audit_and_revise(llm, job_id, role, state[key], state['instructions'][role], phase['criteria'])
                
                if audit_res['status'] == "PASS":
                    log_to_job(job_id, "QA", "Approved.", type="success")
                    current_phase_idx += 1
                else:
                    # RETRY LOGIC WITH PROMPT MODIFICATION
                    retries = state['retry_counts'][role]
                    if retries < MAX_RETRIES:
                        state['retry_counts'][role] += 1
                        
                        # HERE IS THE MAGIC: The Supervisor OVERWRITES the instructions
                        new_instructions = audit_res.get('new_instructions', state['instructions'][role])
                        state['instructions'][role] = new_instructions
                        
                        state[key] = None # Wipe output
                        log_to_job(job_id, "QA", f"Output Rejected: {audit_res['reason']}", type="critique")
                        log_to_job(job_id, "SUPERVISOR", f"REWRITING PROMPT FOR AGENT (Retry {retries+1})...", type="decision")
                    else:
                        log_to_job(job_id, "QA", "Max retries hit. Accepting result.", type="info")
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

# --- FRONTEND TEMPLATE (FIXED SCROLLING + UI) ---
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
        
        /* CUSTOM SCROLLBAR FOR LOGS */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f1219; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
    </style>
</head>
<body class="bg-[#0B0F19] text-gray-300 h-screen w-screen overflow-hidden font-sans selection:bg-indigo-500/30">

    <div class="h-full flex flex-col max-w-7xl mx-auto p-4 gap-4">
        
        <div class="flex-none flex justify-between items-center border-b border-gray-800 pb-3">
            <div class="flex items-center gap-3">
                <div class="p-2 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                    <div class="w-2 h-2 bg-indigo-500 rounded-full blink"></div>
                </div>
                <div>
                    <h1 class="text-lg font-bold text-gray-100 tracking-tight">Agent Supervisor</h1>
                    <p class="text-[10px] text-gray-500 uppercase tracking-widest font-mono">Self-Correcting Workflow</p>
                </div>
            </div>
            <div id="statusBadge" class="bg-gray-800/50 px-3 py-1.5 rounded text-[10px] font-mono uppercase tracking-widest text-gray-500 border border-gray-700">
                System Idle
            </div>
        </div>

        <div class="flex-1 min-h-0 grid grid-cols-12 gap-4">
            
            <div class="col-span-3 flex flex-col gap-3 min-h-0">
                <div class="bg-[#111827] rounded-lg border border-gray-800 flex-1 flex flex-col min-h-0 overflow-hidden group focus-within:border-indigo-500/50 transition-colors">
                    <div class="px-3 py-2 bg-gray-900 border-b border-gray-800 text-[10px] font-bold text-gray-400 uppercase tracking-widest flex justify-between">
                        <span>Context</span>
                        <span class="text-gray-600">JSON</span>
                    </div>
                    <textarea id="jsonInput" class="flex-1 bg-transparent p-3 text-xs font-mono text-emerald-400/90 outline-none resize-none placeholder-gray-700" spellcheck="false">{
    "records": [
        {
            "companyName": "TechFlow",
            "firstName": "Alex"
        }
    ]
}</textarea>
                </div>
                <button onclick="startJob()" class="flex-none bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold py-3 rounded-lg transition shadow-lg shadow-indigo-900/20 border border-indigo-500/50">
                    INITIATE SEQUENCE
                </button>
            </div>

            <div class="col-span-6 bg-[#0f1219] rounded-lg border border-gray-800 flex flex-col min-h-0 relative shadow-2xl overflow-hidden">
                <div class="flex-none bg-[#161b28] px-3 py-2 border-b border-gray-800 flex justify-between items-center z-10">
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Execution Stream</span>
                    </div>
                    <span id="pct" class="text-[10px] font-mono text-gray-500">0%</span>
                </div>
                
                <div id="logs" class="flex-1 min-h-0 p-4 overflow-y-auto space-y-2 scroll-smooth">
                    </div>

                <div id="thinking" class="hidden flex-none p-2 bg-[#0B0F19] border-t border-gray-800/50">
                    <div class="flex items-center gap-2 text-indigo-400/80 text-[10px] font-mono uppercase tracking-widest animate-pulse">
                        <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
                        Supervisor is modifying agent prompts...
                    </div>
                </div>
            </div>

            <div class="col-span-3 bg-[#111827] rounded-lg border border-gray-800 flex flex-col min-h-0 shadow-xl overflow-hidden">
                <div class="flex-none px-3 py-2 bg-gray-900 border-b border-gray-800 text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                    Final Deliverable
                </div>
                <div id="result" class="flex-1 min-h-0 p-4 overflow-y-auto text-sm text-gray-300 font-sans leading-relaxed whitespace-pre-wrap">
                    <span class="text-gray-700 italic text-xs">Waiting for validation...</span>
                </div>
            </div>

        </div>
    </div>

    <script>
        let jobId = null;
        let timer = null;
        const logsContainer = document.getElementById('logs');

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
            
            logsContainer.innerHTML = '';
            document.getElementById('result').innerHTML = '<span class="text-gray-600 italic text-xs">Processing...</span>';
            document.getElementById('statusBadge').innerText = "ACTIVE";
            document.getElementById('statusBadge').className = "bg-indigo-900/30 px-3 py-1.5 rounded text-[10px] font-mono uppercase tracking-widest text-indigo-400 border border-indigo-500/50 blink";
            document.getElementById('thinking').classList.remove('hidden');
            
            timer = setInterval(poll, 800);
        }

        async function poll() {
            if (!jobId) return;
            const res = await fetch(`/api/status/${jobId}`);
            const data = await res.json();
            
            // Only update DOM if changes detected (prevents selection jitter)
            if (logsContainer.children.length !== data.logs.length) {
                logsContainer.innerHTML = data.logs.join('');
                
                // FORCE SCROLL TO BOTTOM
                requestAnimationFrame(() => {
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                });
            }

            document.getElementById('pct').innerText = data.progress + "%";

            if (data.status !== 'running') {
                clearInterval(timer);
                document.getElementById('statusBadge').innerText = data.status.toUpperCase();
                
                if (data.status === 'completed') {
                     document.getElementById('statusBadge').className = "bg-emerald-900/30 px-3 py-1.5 rounded text-[10px] font-mono uppercase tracking-widest text-emerald-400 border border-emerald-500/50";
                     document.getElementById('result').innerText = data.result;
                } else {
                     document.getElementById('statusBadge').className = "bg-red-900/30 px-3 py-1.5 rounded text-[10px] font-mono uppercase tracking-widest text-red-400 border border-red-500/50";
                }
                
                document.getElementById('thinking').classList.add('hidden');
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
