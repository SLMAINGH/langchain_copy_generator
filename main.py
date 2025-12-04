import os
import uuid
import time
import json
import threading
import requests
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

# --- LANGCHAIN IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

app = Flask(__name__)

# --- IN-MEMORY STATE ---
jobs = {}

# --- CONFIGURATION ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY") 

# --- HELPER: LOGGING ---
def log_to_job(job_id, message, is_thought=False):
    """
    Appends messages to the log. 
    If is_thought=True, it adds visual formatting for the 'brain dump'.
    """
    timestamp = datetime.now().strftime('%H:%M:%S')
    if is_thought:
        formatted_msg = f"\n[{timestamp}] 🧠 THINKING:\n{message}\n"
    else:
        formatted_msg = f"[{timestamp}] {message}"
    
    if job_id in jobs:
        jobs[job_id]['logs'].append(formatted_msg)

# --- STEP 1: COMPANY RESEARCH (Perplexity) ---
def step_company_research(job_id, company_name):
    log_to_job(job_id, f"🚀 Contacting Perplexity for live intel on {company_name}...")
    
    if not PERPLEXITY_API_KEY:
        msg = "⚠️ Perplexity Key missing. Using generic knowledge."
        log_to_job(job_id, msg)
        return f"Generic analysis of {company_name} (API Key missing)."

    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a senior market analyst. Be concise but specific."
            },
            {
                "role": "user",
                "content": f"Research {company_name}. Return a summary of: 1. Recent Strategic Shifts (last 6 months). 2. Key Financial Challenges. 3. Current tech focus areas."
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        log_to_job(job_id, content, is_thought=True) # Log the full research
        return content
    except Exception as e:
        log_to_job(job_id, f"❌ Perplexity Error: {e}")
        return f"Error querying Perplexity."

# --- STEP 2: ACCOUNT MAP ---
def step_account_map(llm, job_id, company_research, prospects_json):
    log_to_job(job_id, "🗺️ Mapping prospects to company challenges...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Strategic Account Director.
        
        COMPANY INTELLIGENCE:
        {research}
        
        PROSPECT LIST:
        {prospects}
        
        Task: Select the top 1-2 people who can INFLUENCE the challenges found in the research.
        Explain the 'Why' for each person based on their specific title/bio.
        
        Return JSON with key 'selected_contacts'.
        """
    )
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"research": company_research, "prospects": json.dumps(prospects_json)})
    
    # Log the thought process
    log_to_job(job_id, json.dumps(result, indent=2), is_thought=True)
    return result

# --- STEP 3: STRICT SIGNAL ALIGNMENT (Anti-Hallucination) ---
def step_signal_analysis(llm, job_id, account_map, prospects_full_data):
    """
    Instead of searching for new posts (which risks hallucination if no tool is present),
    we strictly analyze the intersection of the User's Real Bio and the Company's Real Research.
    """
    log_to_job(job_id, "📡 Analying strict alignment signals (Anti-Hallucination Mode)...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Fact-Based Researcher.
        
        TARGETS: {account_map}
        FULL PROFILE DATA: {prospects_data}
        
        Task: For each selected contact, find a 'Hard Signal' of alignment.
        
        RULES:
        1. DO NOT invent LinkedIn posts or webinars that didn't happen.
        2. DO NOT say "They recently posted about X" unless it is in the data.
        3. DO find the specific intersection between their STATED bio/summary and the company's research.
        
        Output the "Signal Strategy" for each person.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"account_map": str(account_map), "prospects_data": str(prospects_full_data)})
    
    log_to_job(job_id, result, is_thought=True)
    return result

# --- STEP 4: OUTREACH STRATEGY ---
def step_outreach_strategy(llm, job_id, signals):
    log_to_job(job_id, "♟️ Devising LinkedIn strategy...")
    
    prompt = ChatPromptTemplate.from_template(
        """Based on these verified signals: 
        {signals}
        
        Create a 'Hook Strategy' for a LinkedIn approach. 
        How do we bridge the gap between their specific responsibility and our value?
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"signals": signals})
    
    log_to_job(job_id, result, is_thought=True)
    return result

# --- STEP 5: LINKEDIN WRITING ---
def step_write_linkedin(llm, job_id, strategy, account_map):
    log_to_job(job_id, "✍️ Drafting LinkedIn Messages...")
    
    prompt = ChatPromptTemplate.from_template(
        """
        CONTEXT: {strategy}
        TARGETS: {account_map}
        
        Task: Write 2 LinkedIn messages for each target.
        
        Format 1: CONNECTION REQUEST
        - STRICT LIMIT: Under 300 characters (including spaces).
        - No fluff. Contextual.
        
        Format 2: INMAIL / FOLLOW-UP
        - Subject Line (if InMail)
        - Body: Casual, mobile-friendly, professional.
        
        Output plainly formatted text.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"strategy": strategy, "account_map": str(account_map)})
    
    log_to_job(job_id, result, is_thought=True)
    return result

# --- WORKER THREAD ---
def process_workflow(job_id, input_data):
    job = jobs[job_id]
    
    try:
        user_api_key = input_data.get('openai_api_key')
        api_key = user_api_key if user_api_key else OPENAI_API_KEY
        
        if not api_key:
            log_to_job(job_id, "❌ Critical: No OpenAI API Key found.")
            job['status'] = "failed"
            return

        llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.0) # Temp 0 reduces hallucinations
        
        records = input_data['input_json'].get('records', [])
        if not records:
             raise ValueError("No records found in JSON")
             
        company_name = records[0].get('companyName', "Unknown Company")

        # 1. Research
        job['step'] = 1
        research_output = step_company_research(job_id, company_name)
        
        # 2. Map
        job['step'] = 2
        account_map = step_account_map(llm, job_id, research_output, records)

        # 3. Signals (Strict)
        job['step'] = 3
        signals = step_signal_analysis(llm, job_id, account_map, records)

        # 4. Strategy
        job['step'] = 4
        strategy = step_outreach_strategy(llm, job_id, signals)

        # 5. Writing
        job['step'] = 5
        final_copy = step_write_linkedin(llm, job_id, strategy, account_map)

        job['result'] = final_copy
        job['status'] = "completed"
        job['progress'] = 100
        log_to_job(job_id, "✅ Workflow Finished Successfully.")

    except Exception as e:
        job['status'] = "failed"
        log_to_job(job_id, f"❌ Workflow Crash: {str(e)}")
        print(f"Error in job {job_id}: {e}")

# --- API ROUTES ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/start', methods=['POST'])
def start_job():
    data = request.json
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "logs": [],
        "result": None,
        "step": 0,
        "total_steps": 5,
        "progress": 0
    }
    
    thread = threading.Thread(target=process_workflow, args=(job_id, data))
    thread.start()
    
    return jsonify({"job_id": job_id, "message": "Workflow started"})

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    if job['status'] == "running":
        job['progress'] = int((job['step'] / job['total_steps']) * 100)
        
    return jsonify(job)

@app.route('/health')
def health():
    return "OK", 200

# --- FRONTEND TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LinkedIn Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .log-entry { 
            font-family: 'Courier New', monospace; 
            font-size: 0.85em; 
            border-bottom: 1px solid #333; 
            padding: 8px 0; 
            white-space: pre-wrap; /* Keeps formatting */
        }
        .blink { animation: blinker 1s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1a1a1a; }
        ::-webkit-scrollbar-thumb { background: #4a4a4a; border-radius: 4px; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 p-6 font-sans">

    <div class="max-w-6xl mx-auto">
        <div class="flex items-center justify-between mb-8">
            <h1 class="text-3xl font-bold text-blue-400">🔗 Autonomous LinkedIn Agent</h1>
            <div id="statusBadge" class="bg-gray-800 text-gray-400 text-xs px-3 py-1 rounded-full uppercase tracking-wider">Idle</div>
        </div>
        
        <div id="inputSection" class="bg-gray-800 p-6 rounded-lg shadow-lg mb-8 border border-gray-700">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                    <label class="block text-xs font-bold text-gray-400 uppercase mb-1">OpenAI Key</label>
                    <input type="password" id="apiKey" class="w-full bg-gray-900 border border-gray-600 rounded p-2 text-sm focus:border-blue-500 outline-none transition">
                </div>
            </div>
            
            <label class="block text-xs font-bold text-gray-400 uppercase mb-1">Input Data (JSON)</label>
            <textarea id="jsonInput" rows="6" class="w-full bg-gray-900 border border-gray-600 rounded p-3 font-mono text-xs text-green-400 focus:border-blue-500 outline-none" placeholder='{ "records": [...] }'></textarea>
            
            <button onclick="startJob()" class="mt-4 bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-6 rounded transition w-full md:w-auto">
                Launch Agent
            </button>
        </div>

        <div id="trackingSection" class="hidden grid grid-cols-1 lg:grid-cols-2 gap-6 h-[600px]">
            
            <div class="flex flex-col bg-black rounded-lg border border-gray-700 shadow-xl overflow-hidden">
                <div class="bg-gray-800 px-4 py-2 border-b border-gray-700 flex justify-between items-center">
                    <span class="text-xs font-bold text-gray-400 uppercase">⚡ Agent Chain of Thought</span>
                    <span id="pctText" class="text-xs text-blue-400">0%</span>
                </div>
                <div id="logsContainer" class="flex-1 p-4 overflow-y-auto text-green-500 font-mono text-xs leading-relaxed">
                    </div>
            </div>
            
            <div class="flex flex-col bg-white rounded-lg border border-gray-700 shadow-xl overflow-hidden">
                <div class="bg-gray-200 px-4 py-2 border-b border-gray-300">
                    <span class="text-xs font-bold text-gray-600 uppercase">📝 Generated Messages</span>
                </div>
                <div id="resultContainer" class="flex-1 p-6 overflow-y-auto text-gray-800 font-sans text-sm whitespace-pre-wrap">
                    <p class="text-gray-400 italic text-center mt-20">Output will appear here upon completion...</p>
                </div>
            </div>

        </div>
    </div>

    <script>
        let currentJobId = null;
        let pollInterval = null;

        async function startJob() {
            const apiKey = document.getElementById('apiKey').value;
            const jsonText = document.getElementById('jsonInput').value;
            
            try {
                JSON.parse(jsonText); // Validate JSON
                
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        openai_api_key: apiKey,
                        input_json: JSON.parse(jsonText)
                    })
                });
                
                const data = await response.json();
                currentJobId = data.job_id;
                
                document.getElementById('inputSection').classList.add('opacity-50', 'pointer-events-none');
                document.getElementById('trackingSection').classList.remove('hidden');
                document.getElementById('statusBadge').innerText = "RUNNING";
                document.getElementById('statusBadge').className = "bg-blue-900 text-blue-200 text-xs px-3 py-1 rounded-full uppercase tracking-wider blink";
                
                pollInterval = setInterval(checkStatus, 1000); // Poll every 1s
                
            } catch (e) {
                alert("Invalid JSON data. Please check syntax.");
            }
        }

        async function checkStatus() {
            if (!currentJobId) return;
            
            const response = await fetch(`/api/status/${currentJobId}`);
            const data = await response.json();
            
            // Update percentage
            document.getElementById('pctText').innerText = data.progress + "%";
            
            // Render Logs (The "Brain")
            // We join them differently to handle the newlines in thoughts
            const logsContainer = document.getElementById('logsContainer');
            logsContainer.innerHTML = data.logs.join(""); 
            logsContainer.scrollTop = logsContainer.scrollHeight;

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                document.getElementById('resultContainer').innerText = data.result;
                document.getElementById('statusBadge').innerText = "COMPLETE";
                document.getElementById('statusBadge').className = "bg-green-900 text-green-200 text-xs px-3 py-1 rounded-full uppercase tracking-wider";
                document.getElementById('inputSection').classList.remove('opacity-50', 'pointer-events-none');
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                document.getElementById('statusBadge').innerText = "FAILED";
                document.getElementById('statusBadge').className = "bg-red-900 text-red-200 text-xs px-3 py-1 rounded-full uppercase tracking-wider";
                document.getElementById('inputSection').classList.remove('opacity-50', 'pointer-events-none');
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
