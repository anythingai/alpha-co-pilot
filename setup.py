#!/usr/bin/env python3
"""
Alpha Co-Pilot Setup Script
Quick setup for the hackathon demo
"""

import os
import subprocess
import sys

def create_env_file():
    """Create a .env file with required environment variables"""
    env_content = """# Alpha Co-Pilot Environment Variables
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini

# Create a Discord webhook URL from your test Discord server
# Go to Server Settings > Integrations > Webhooks > New Webhook
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here

# Flask secret key for sessions
FLASK_SECRET_KEY=alpha_copilot_hackathon_2025_secret_key

# Optional: Whop API credentials (for real OAuth integration)
WHOP_CLIENT_ID=your_whop_client_id
WHOP_CLIENT_SECRET=your_whop_client_secret
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ Created .env file - Please update it with your API keys!")
    else:
        print("⚠️  .env file already exists - skipping creation")

def create_directory_structure():
    """Create necessary directories"""
    directories = ['templates', 'static', 'static/css', 'static/js']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def save_html_templates():
    """Save the HTML templates to files"""
    
    # Dashboard template (from previous artifact)
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha Co-Pilot - Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <div class="min-h-screen">
        <!-- Header -->
        <header class="gradient-bg shadow-lg">
            <div class="max-w-6xl mx-auto px-4 py-6">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                            <span class="text-purple-600 font-bold text-xl">α</span>
                        </div>
                        <h1 class="text-2xl font-bold text-white">Alpha Co-Pilot</h1>
                    </div>
                    <div class="text-sm text-gray-200">
                        Connected to Whop • Beta User
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-4xl mx-auto px-4 py-12">
            <!-- Hero Section -->
            <div class="text-center mb-12">
                <h2 class="text-4xl font-bold text-white mb-4">From Data to Alpha in Seconds</h2>
                <p class="text-xl text-gray-300 mb-8">Enter any crypto topic and get instant, professional analysis for your community</p>
            </div>

            <!-- Input Section -->
            <div class="bg-gray-800 rounded-xl shadow-2xl p-8 mb-8">
                <div class="space-y-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">What alpha do you want to generate?</label>
                        <div class="relative">
                            <input 
                                type="text" 
                                id="queryInput"
                                placeholder="e.g., Analyze trending AI coins on Solana"
                                class="w-full px-4 py-4 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent text-lg"
                            />
                        </div>
                    </div>
                    <button 
                        id="generateBtn"
                        class="w-full gradient-bg text-white font-semibold py-4 px-6 rounded-lg hover:shadow-lg transform hover:scale-105 transition-all duration-200 text-lg"
                    >
                        Generate Alpha ✨
                    </button>
                </div>
            </div>

            <!-- Loading State -->
            <div id="loadingState" class="hidden text-center py-12">
                <div class="loading-spinner mx-auto mb-4"></div>
                <p class="text-gray-300 text-lg">Analyzing market data and generating alpha...</p>
            </div>

            <!-- Results Section -->
            <div id="resultsSection" class="hidden fade-in">
                <div class="bg-gray-800 rounded-xl shadow-2xl p-8">
                    <div class="flex items-center justify-between mb-6">
                        <h3 class="text-2xl font-bold text-white">Fresh Alpha Generated</h3>
                        <div class="flex space-x-4">
                            <button 
                                id="shareBtn"
                                class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200"
                            >
                                Share to Discord 🚀
                            </button>
                            <button 
                                id="copyBtn"
                                class="bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200"
                            >
                                Copy Text
                            </button>
                        </div>
                    </div>
                    
                    <!-- Analysis Output -->
                    <div class="bg-gray-900 rounded-lg p-6 mb-6">
                        <div id="analysisContent" class="text-gray-100 whitespace-pre-wrap font-mono text-sm leading-relaxed"></div>
                    </div>
                    
                    <!-- Coin Data Cards -->
                    <div id="coinDataSection" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <!-- Coin cards will be inserted here dynamically -->
                    </div>
                </div>
            </div>

            <!-- Error State -->
            <div id="errorState" class="hidden bg-red-900 border border-red-700 rounded-lg p-6 text-center">
                <p class="text-red-300" id="errorMessage"></p>
            </div>
        </main>

        <!-- Footer -->
        <footer class="text-center py-8 text-gray-500">
            <p>Built for the Whop App Store • Powered by Real-Time Data</p>
        </footer>
    </div>

    <script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
</body>
</html>"""

    # Save templates with UTF-8 encoding
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print("✅ Created templates/dashboard.html")
    


def create_dashboard_js():
    """Create the dashboard JavaScript file"""
    js_content = """const queryInput = document.getElementById('queryInput');
const generateBtn = document.getElementById('generateBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const errorState = document.getElementById('errorState');
const analysisContent = document.getElementById('analysisContent');
const coinDataSection = document.getElementById('coinDataSection');
const shareBtn = document.getElementById('shareBtn');
const copyBtn = document.getElementById('copyBtn');

let currentAnalysis = '';

// Sample queries for demo
const sampleQueries = [
    "Analyze trending AI coins on Solana",
    "What are the top 3 DePIN tokens to watch?",
    "Trending meme coins with high volume",
    "Best Layer 2 tokens for this week"
];

// Add sample query on page load
window.addEventListener('load', () => {
    const randomQuery = sampleQueries[Math.floor(Math.random() * sampleQueries.length)];
    queryInput.placeholder = randomQuery;
});

generateBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a query');
        return;
    }

    showLoading();
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });

        const data = await response.json();
        
        if (data.success) {
            showResults(data);
        } else {
            showError(data.error || 'Failed to generate alpha');
        }
    } catch (error) {
        showError('Network error. Please try again.');
        console.error('Error:', error);
    }
});

shareBtn.addEventListener('click', async () => {
    if (!currentAnalysis) return;

    try {
        shareBtn.disabled = true;
        shareBtn.textContent = 'Sharing...';
        
        const response = await fetch('/api/share', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                content: currentAnalysis,
                title: '🚀 Fresh Alpha Alert' 
            })
        });

        const data = await response.json();
        
        if (data.success) {
            shareBtn.textContent = 'Shared! ✅';
            shareBtn.className = 'bg-green-600 text-white font-medium py-2 px-6 rounded-lg';
        } else {
            shareBtn.textContent = 'Share Failed';
            shareBtn.className = 'bg-red-600 text-white font-medium py-2 px-6 rounded-lg';
        }
        
        setTimeout(() => {
            shareBtn.disabled = false;
            shareBtn.textContent = 'Share to Discord 🚀';
            shareBtn.className = 'bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors duration-200';
        }, 3000);
    } catch (error) {
        console.error('Share error:', error);
        shareBtn.textContent = 'Share Failed';
        shareBtn.className = 'bg-red-600 text-white font-medium py-2 px-6 rounded-lg';
    }
});

copyBtn.addEventListener('click', () => {
    if (currentAnalysis) {
        navigator.clipboard.writeText(currentAnalysis).then(() => {
            copyBtn.textContent = 'Copied! ✅';
            setTimeout(() => {
                copyBtn.textContent = 'Copy Text';
            }, 2000);
        });
    }
});

// Enter key support
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        generateBtn.click();
    }
});

function showLoading() {
    hideAllStates();
    loadingState.classList.remove('hidden');
}

function showResults(data) {
    hideAllStates();
    
    currentAnalysis = data.analysis;
    analysisContent.textContent = data.analysis;
    
    // Display coin data cards
    coinDataSection.innerHTML = '';
    if (data.coin_data && data.coin_data.length > 0) {
        data.coin_data.forEach(coin => {
            const coinCard = createCoinCard(coin);
            coinDataSection.appendChild(coinCard);
        });
    }
    
    resultsSection.classList.remove('hidden');
}

function showError(message) {
    hideAllStates();
    document.getElementById('errorMessage').textContent = message;
    errorState.classList.remove('hidden');
}

function hideAllStates() {
    loadingState.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorState.classList.add('hidden');
}

function createCoinCard(coin) {
    const card = document.createElement('div');
    const priceChange = coin.price_change_24h || 0;
    const changeColor = priceChange >= 0 ? 'text-green-400' : 'text-red-400';
    const changeIcon = priceChange >= 0 ? '↗️' : '↘️';
    
    card.className = 'bg-gray-700 rounded-lg p-4';
    card.innerHTML = `
        <div class="text-center">
            <h4 class="font-bold text-white text-lg">${coin.symbol}</h4>
            <p class="text-gray-400 text-sm mb-2">${coin.name}</p>
            <div class="text-xl font-bold text-white mb-1">
                $${coin.current_price ? coin.current_price.toLocaleString() : 'N/A'}
            </div>
            <div class="${changeColor} text-sm font-medium">
                ${changeIcon} ${priceChange ? priceChange.toFixed(2) : '0.00'}%
            </div>
        </div>
    `;
    return card;
}"""
    
    with open('static/js/dashboard.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    print("✅ Created static/js/dashboard.js")

def install_requirements():
    """Install Python requirements"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Installed Python requirements")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please run: pip install -r requirements.txt")

def print_setup_instructions():
    """Print final setup instructions"""
    instructions = """
🚀 ALPHA CO-PILOT SETUP COMPLETE! 🚀

Next steps:

3. CONFIGURE ENVIRONMENT:
   - Add your Azure OpenAI API key and endpoint to .env
   - Create Discord webhook: Server Settings > Integrations > Webhooks
   - Update the .env file with your keys

2. RUN THE APPLICATION:
   python app.py

3. OPEN YOUR BROWSER:
   http://localhost:5000/login

4. FOR THE DEMO VIDEO:
   - Start at /login (shows Whop integration)
   - Click "Continue with Whop" 
   - Try query: "Analyze trending AI coins on Solana"
   - Show the generate and share functionality

5. HACKATHON DEMO TIPS:
   - Have your Discord webhook ready for live sharing
   - Test with different crypto queries
   - Show the real-time data integration
   - Emphasize the 3-5 hour time savings for creators

Good luck with your hackathon! 🏆
"""
    print(instructions)

if __name__ == "__main__":
    print("🔧 Setting up Alpha Co-Pilot...")
    
    create_directory_structure()
    create_env_file()
    save_html_templates()
    create_dashboard_js()
    install_requirements()
    
    print_setup_instructions()