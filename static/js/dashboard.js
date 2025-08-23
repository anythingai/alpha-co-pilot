const queryInput = document.getElementById('queryInput');
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
    // Render Markdown-formatted analysis
    analysisContent.innerHTML = marked.parse(data.analysis);
    
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
}