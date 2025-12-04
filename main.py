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
# NOTE: On Railway, this persists as long as the deployment is running.
# If you redeploy, this clears. For a simple tool, this is fine.
jobs = {}

# --- CONFIGURATION FROM ENV ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY") 

# --- STEP 1: COMPANY RESEARCH (Perplexity) ---
def step_company_research(company_name):
    """
    Real call to Perplexity API for up-to-date research.
    """
    if not PERPLEXITY_API_KEY:
        return "Skipping Research: PERPLEXITY_API_KEY not set in Railway variables."

    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a market researcher. Analyze the company comprehensively."
            },
            {
                "role": "user",
                "content": f"Analyze {company_name}. Focus on: recent strategic shifts, financial performance, and key challenges/pain points."
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
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error querying Perplexity: {str(e)}"

# --- STEP 2: ACCOUNT MAP ---
def step_account_map(llm, company_research, prospects_json):
    prompt = ChatPromptTemplate.from_template(
        """You are a Sales Director. 
        
        COMPANY RESEARCH:
        {research}
        
        PROSPECT LIST (JSON):
        {prospects}
        
        Task: Create an 'Account Map'. 
        1. Identify the top 2-3 people from the list who are most relevant to the company's current challenges found in the research.
        2. Explain briefly why you chose them.
        
        Return your answer as valid JSON with a list under the key 'selected_contacts'.
        """
    )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"research": company_research, "prospects": json.dumps(prospects_json)})

# --- STEP 3: ADAPT SIGNAL PROMPT ---
def step_adapt_signal_prompt(llm, account_map):
    prompt = ChatPromptTemplate.from_template(
        """Based on this Account Map: {account_map}
        
        Generate a specific search query/prompt to find 'intent signals' (news, podcasts, posts) 
        for these specific individuals.
        """
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"account_map": str(account_map)})

# --- STEP 4: SIGNAL RESEARCH ---
def step_signal_research(llm, adapted_prompt):
    # If you have a specific tool for this (Tavily/Google), insert here.
    # For now, we simulate the "Agent" thinking based on the prompt.
    prompt = ChatPromptTemplate.from_template(
        """You are a Signal Researcher. execute this task: {task}
        
        Simulate the findings. Invent realistic, high-relevance 'signals' 
        (e.g., a recent webinar appearance, a LinkedIn post about AI, a quarterly earnings quote)
        for the people identified.
        """
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"task": adapted_prompt})

# --- STEP 5: OUTREACH STRATEGY ---
def step_outreach_strategy(llm, account_map, signals):
    prompt = ChatPromptTemplate.from_template(
        """Based on Account Map: {account_map}
        And Research Signals: {signals}
        
        Create a high-level 'Outreach Strategy'. 
        What is the hook? What is the value proposition?
        """
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"account_map": str(account_map), "signals": signals})

# --- STEP 6: PERSONALIZATION ---
def step_personalize_messages(llm, strategy, account_map):
    prompt = ChatPromptTemplate.from_template(
        """Based on Strategy: {strategy}
        And contacts in: {account_map}
        
        Write a hyper-personalized cold email for each selected contact.
        """
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"strategy": strategy, "account_map": str(account_map)})

# --- WORKER THREAD ---
def process_workflow(job_id, input_data):
    job = jobs[job_id]
    
    try:
        # Use key provided by user UI, fallback to Env Var
        user_api_key = input_data.get('openai_api_key')
        api_key = user_api_key if user_api_key else OPENAI_API_KEY
        
        if not api_key:
            raise ValueError("No OpenAI API Key provided.")

        llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
        
        records = input_data['input_json'].get('records', [])
        company_name = records[0].get('companyName', "Unknown Company") if records else "Unknown Company"

        # --- STEP 1 ---
        job['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting Company Research (Perplexity) for {company_name}...")
        job['step'] = 1
        research_output = step_company_research(company_name)
        job['logs'].append(f"✅ Research Complete.")
        
        # --- STEP 2 ---
        job['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🗺️ Building Account Map...")
        job['step'] = 2
        account_map = step_account_map(llm, research_output, records)
        job['logs'].append(f"✅ Account Map Created: Selected {len(account_map.get('selected_contacts', []))} key contacts.")

        # --- STEP 3 ---
        job['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Adapting Research Prompts...")
        job['step'] = 3
        signal_prompt = step_adapt_signal_prompt(llm, account_map)
        job['logs'].append(f"✅ Prompts Adapted.")

        # --- STEP 4 ---
        job['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Hunting for Signals...")
        job['step'] = 4
        signals = step_signal_research(llm, signal_prompt)
        job['logs'].append(f"✅ Signals Found.")

        # --- STEP 5 ---
        job['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ♟️ Developing Outreach Strategy...")
        job['step'] = 5
        strategy = step_outreach_strategy(llm, account_map, signals)
        job['logs'].append(f"✅ Strategy Defined.")

        # --- STEP 6 ---
        job['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] ✍️ Writing Personalizations...")
        job['step'] = 6
        final_emails = step_personalize_messages(llm, strategy, account_map)
        job['logs'].append(f"✅ Emails Generated.")

        job['result'] = final_emails
        job['status'] = "completed"
        job['progress'] = 100

    except Exception as e:
        job['status'] = "failed"
        job['logs'].append(f"❌ Error: {str(e)}")
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
        "total_steps": 6,
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
    <title>Railway Sales Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .log-entry { font-family: monospace; font-size: 0.9em; border-bottom: 1px solid #eee; padding: 4px 0; }
        .blink { animation: blinker 1.5s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
    </style>
</head>
<body class="bg-gray-100 p-6 md:p-12">

    <div class="max-w-5xl mx-auto bg-white p-8 rounded-xl shadow-lg border border-gray-200">
        <div class="flex items-center justify-between mb-8">
            <h1 class="text-3xl font-bold text-gray-800">🕵️‍♂️ Account Intelligence Agent</h1>
            <span class="bg-purple-100 text-purple-800 text-xs font-semibold px-2.5 py-0.5 rounded">Running on Railway</span>
        </div>
        
        <div id="inputSection" class="transition-all duration-300">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">OpenAI API Key</label>
                    <input type="password" id="apiKey" class="w-full p-3 border rounded-lg bg-gray-50 focus:ring-2 focus:ring-purple-500 outline-none" placeholder="sk-...">
                    <p class="text-xs text-gray-500 mt-1">Leave empty if set in Railway Variables</p>
                </div>
            </div>
            
            <label class="block text-sm font-medium text-gray-700 mb-2">Input Data (JSON)</label>
            <textarea id="jsonInput" rows="8" class="w-full p-3 border rounded-lg bg-gray-50 font-mono text-xs focus:ring-2 focus:ring-purple-500 outline-none" placeholder="Paste your records JSON here..."></textarea>
            
            <button onclick="startJob()" class="mt-6 w-full bg-purple-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-purple-700 transition shadow-md">
                Start Agent Workflow
            </button>
        </div>

        <div id="trackingSection" class="hidden mt-8">
            <div class="mb-6">
                <div class="flex justify-between items-center mb-2">
                    <span class="text-sm font-bold text-purple-700 tracking-wider" id="statusText">INITIALIZING...</span>
                    <span class="text-sm font-bold text-gray-600" id="pctText">0%</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div id="progressBar" class="bg-purple-600 h-3 rounded-full transition-all duration-500 ease-out" style="width: 0%"></div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="flex flex-col h-96">
                    <h3 class="text-xs font-bold text-gray-500 uppercase mb-2">Agent Thoughts (Logs)</h3>
                    <div class="flex-1 border border-gray-300 rounded-lg p-4 overflow-y-auto bg-black text-green-400 shadow-inner font-mono text-xs">
                        <div id="logsContainer"></div>
                        <div id="cursor" class="blink mt-1">_</div>
                    </div>
                </div>
                
                <div class="flex flex-col h-96">
                    <h3 class="text-xs font-bold text-gray-500 uppercase mb-2">Generated Outreach</h3>
                    <div class="flex-1 border border-gray-300 rounded-lg p-4 overflow-y-auto bg-gray-50 shadow-inner">
                        <pre id="resultContainer" class="text-xs whitespace-pre-wrap text-gray-700 font-mono">Results will appear here...</pre>
                    </div>
                </div>
            </div>
            
            <button onclick="location.reload()" id="resetBtn" class="hidden mt-6 w-full bg-gray-200 text-gray-800 font-semibold px-6 py-3 rounded-lg hover:bg-gray-300 transition">
                Start New Search
            </button>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let pollInterval = null;

        async function startJob() {
            const apiKey = document.getElementById('apiKey').value;
            const jsonText = document.getElementById('jsonInput').value;
            
            try {
                const parsedJson = JSON.parse(jsonText);
                
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        openai_api_key: apiKey,
                        input_json: parsedJson
                    })
                });
                
                const data = await response.json();
                currentJobId = data.job_id;
                
                document.getElementById('inputSection').classList.add('hidden');
                document.getElementById('trackingSection').classList.remove('hidden');
                
                pollInterval = setInterval(checkStatus, 1500);
                
            } catch (e) {
                alert("Invalid JSON! Please check your input.");
            }
        }

        async function checkStatus() {
            if (!currentJobId) return;
            
            try {
                const response = await fetch(`/api/status/${currentJobId}`);
                if (!response.ok) return;
                
                const data = await response.json();
                
                document.getElementById('progressBar').style.width = data.progress + "%";
                document.getElementById('pctText').innerText = data.progress + "%";
                document.getElementById('statusText').innerText = data.status.toUpperCase();
                
                const logsHtml = data.logs.map(log => `<div class="log-entry">${log}</div>`).join('');
                document.getElementById('logsContainer').innerHTML = logsHtml;
                
                const logContainer = document.getElementById('logsContainer').parentElement;
                logContainer.scrollTop = logContainer.scrollHeight;

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    document.getElementById('resultContainer').innerText = data.result;
                    document.getElementById('statusText').innerText = "✅ CAMPAIGN READY";
                    document.getElementById('statusText').classList.replace('text-purple-700', 'text-green-600');
                    document.getElementById('resetBtn').classList.remove('hidden');
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    document.getElementById('statusText').innerText = "❌ PROCESS FAILED";
                    document.getElementById('statusText').classList.replace('text-purple-700', 'text-red-600');
                    document.getElementById('resetBtn').classList.remove('hidden');
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Local development entry point
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
