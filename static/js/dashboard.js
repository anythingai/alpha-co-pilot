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

    // Update button state
    generateBtn.disabled = true;
    const originalHTML = generateBtn.innerHTML;
    generateBtn.innerHTML = `
        <span class="flex items-center justify-center space-x-2">
            <div class="loading-spinner w-4 h-4"></div>
            <span>Generating...</span>
        </span>
    `;

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
            showNotification('Analysis generated successfully!', 'success');
        } else {
            showError(data.error || 'Failed to generate alpha');
        }
    } catch (error) {
        showError('Network error. Please try again.');
        console.error('Error:', error);
    } finally {
        // Reset button state
        generateBtn.disabled = false;
        generateBtn.innerHTML = originalHTML;
    }
});

shareBtn.addEventListener('click', async () => {
    if (!currentAnalysis) return;

    const originalHTML = shareBtn.innerHTML;
    try {
        shareBtn.disabled = true;
        shareBtn.innerHTML = `
            <div class="loading-spinner w-4 h-4 mr-2"></div>
            Sharing...
        `;
        
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
            shareBtn.innerHTML = `
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                Shared!
            `;
            shareBtn.classList.add('bg-green-600');
            shareBtn.classList.remove('bg-primary');
            showNotification('Successfully shared to Discord!', 'success');
        } else {
            shareBtn.innerHTML = 'Share Failed';
            shareBtn.classList.add('bg-red-600');
            shareBtn.classList.remove('bg-primary');
            showNotification('Failed to share to Discord', 'error');
        }
        
        setTimeout(() => {
            shareBtn.disabled = false;
            shareBtn.innerHTML = originalHTML;
            shareBtn.classList.remove('bg-green-600', 'bg-red-600');
            shareBtn.classList.add('bg-primary');
        }, 3000);
    } catch (error) {
        console.error('Share error:', error);
        shareBtn.innerHTML = 'Share Failed';
        shareBtn.classList.add('bg-red-600');
        shareBtn.classList.remove('bg-primary');
        showNotification('Network error while sharing', 'error');
        
        setTimeout(() => {
            shareBtn.disabled = false;
            shareBtn.innerHTML = originalHTML;
            shareBtn.classList.remove('bg-red-600');
            shareBtn.classList.add('bg-primary');
        }, 3000);
    }
});

copyBtn.addEventListener('click', async () => {
    if (currentAnalysis) {
        try {
            await navigator.clipboard.writeText(currentAnalysis);
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = `
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                Copied!
            `;
            copyBtn.classList.add('bg-green-600', 'text-white');
            copyBtn.classList.remove('bg-muted', 'text-muted-foreground');
            
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                copyBtn.classList.remove('bg-green-600', 'text-white');
                copyBtn.classList.add('bg-muted', 'text-muted-foreground');
            }, 2000);
            
            showNotification('Analysis copied to clipboard!', 'success');
        } catch (error) {
            showNotification('Failed to copy to clipboard', 'error');
        }
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
    const priceChange = coin.price_change_24h || coin.change_24h_pct || 0;
    const price = coin.current_price || coin.price_usd || 0;
    const symbol = coin.symbol ? coin.symbol.toUpperCase() : 'N/A';
    const name = coin.name || coin.id || 'Unknown';
    
    const changeColor = priceChange >= 0 ? 'text-green-500' : 'text-red-500';
    const changeIcon = priceChange >= 0 ? '↗️' : '↘️';
    const changeBg = priceChange >= 0 ? 'bg-green-500/10' : 'bg-red-500/10';
    
    card.className = 'bg-card border border-border rounded-lg p-6 card-hover';
    card.innerHTML = `
        <div class="space-y-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-muted rounded-full flex items-center justify-center">
                        <span class="text-foreground font-semibold text-sm">${symbol.charAt(0)}</span>
                    </div>
                    <div>
                        <h4 class="font-semibold text-foreground text-base">${symbol}</h4>
                        <p class="text-muted-foreground text-xs">${name}</p>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-lg font-semibold text-foreground">
                        $${price ? formatPrice(price) : 'N/A'}
                    </div>
                </div>
            </div>
            
            <div class="flex items-center justify-between pt-2 border-t border-border">
                <span class="text-muted-foreground text-xs">24h Change</span>
                <div class="flex items-center space-x-2">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${changeBg} ${changeColor}">
                        <span class="mr-1">${changeIcon}</span>
                        ${priceChange ? Math.abs(priceChange).toFixed(2) : '0.00'}%
                    </span>
                </div>
            </div>
        </div>
    `;
    return card;
}

function formatPrice(price) {
    if (price >= 1000000) {
        return (price / 1000000).toFixed(2) + 'M';
    } else if (price >= 1000) {
        return (price / 1000).toFixed(2) + 'K';
    } else if (price >= 1) {
        return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } else {
        return price.toFixed(6);
    }
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    const notification = document.createElement('div');
    notification.className = 'notification fixed top-4 right-4 z-50 max-w-sm';
    
    const bgColor = type === 'success' ? 'bg-green-600' : type === 'error' ? 'bg-red-600' : 'bg-blue-600';
    const icon = type === 'success' ? 
        `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
         </svg>` : 
        type === 'error' ?
        `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
         </svg>` :
        `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
         </svg>`;

    notification.innerHTML = `
        <div class="${bgColor} text-white px-4 py-3 rounded-lg shadow-lg flex items-center space-x-3 slide-up">
            ${icon}
            <span class="text-sm font-medium">${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-white/80 hover:text-white">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }
    }, 4000);
}