// ============================================================================
// POLYGON AMOY TESTNET WEB3 CONFIGURATION
// ============================================================================
const POLYGON_AMOY_CHAIN_ID = 80002;
const CONTRACT_ADDRESS = '0x0FeDbdc549619dF0aad590438840bF4A696C7ACA'; // Deployed contract address

// Multiple RPC URLs for redundancy
const POLYGON_RPC_URLS = [
    'https://rpc-amoy.polygon.technology',
    'https://polygon-amoy.drpc.org',
    'https://polygon-amoy-bor.publicnode.com'
];
const POLYGON_RPC_URL = POLYGON_RPC_URLS[0]; // Primary RPC

// Contract ABI for SignalRegistry
const SIGNAL_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "contentHash", "type": "string"},
            {"internalType": "string", "name": "category", "type": "string"}
        ],
        "name": "registerSignal",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "internalType": "address", "name": "creator", "type": "address"},
            {"indexed": true, "internalType": "string", "name": "contentHash", "type": "string"},
            {"indexed": false, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": false, "internalType": "string", "name": "category", "type": "string"}
        ],
        "name": "SignalRegistered",
        "type": "event"
    },
    {
        "inputs": [{"internalType": "address", "name": "creator", "type": "address"}],
        "name": "getSignalCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
];

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================
let web3Provider = null;
let userAccount = null;
let currentAnalysis = '';

// DOM elements
const queryInput = document.getElementById('queryInput');
const generateBtn = document.getElementById('generateBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const errorState = document.getElementById('errorState');
const analysisContent = document.getElementById('analysisContent');
const coinDataSection = document.getElementById('coinDataSection');
const copyBtn = document.getElementById('copyBtn');
const quickStartTemplates = document.getElementById('quickStartTemplates');

// Web3 elements
const connectWalletBtn = document.getElementById('navConnectWalletBtn');
const lockOnChainBtn = document.getElementById('lockOnChainBtn');
const listMarketplaceBtn = document.getElementById('listMarketplaceBtn');
const creaRewards = document.getElementById('creaRewards');

// Wallet dropdown elements
const walletDropdown = document.getElementById('walletDropdown');
const walletAddressDisplay = document.getElementById('walletAddressDisplay');
const copyAddressBtn = document.getElementById('copyAddressBtn');
const viewOnExplorerBtn = document.getElementById('viewOnExplorerBtn');
const disconnectWalletBtn = document.getElementById('disconnectWalletBtn');

// Pre-filled prompt templates
const promptTemplates = document.querySelectorAll('.prompt-template');

// ============================================================================
// WEB3 FUNCTIONS
// ============================================================================

// Initialize Web3 on page load
async function initWeb3() {
    console.log('🚀 Initializing Web3 for Polygon Amoy...');
    
    if (typeof window.ethereum !== 'undefined') {
        try {
            web3Provider = window.ethereum;
            
            // Check if already connected
            const accounts = await web3Provider.request({ method: 'eth_accounts' });
            if (accounts.length > 0) {
                userAccount = accounts[0];
                await checkNetwork();
                updateWalletUI();
                console.log('✅ Wallet already connected:', userAccount);
            }
            
            // Listen for account changes
            web3Provider.on('accountsChanged', handleAccountsChanged);
            web3Provider.on('chainChanged', handleChainChanged);
            
        } catch (error) {
            console.error('❌ Failed to initialize Web3:', error);
        }
    } else {
        console.log('⚠️ MetaMask not detected');
    }
}

// Connect wallet
async function connectWallet() {
    if (!web3Provider) {
        showNotification('Please install MetaMask to use Web3 features!', 'error');
        return;
    }
    
    try {
        console.log('🔗 Connecting wallet...');
        connectWalletBtn.disabled = true;
        connectWalletBtn.innerHTML = `
            <div class="loading-spinner w-4 h-4 mr-2"></div>
            Connecting...
        `;
        
        // Request account access
        const accounts = await web3Provider.request({ 
            method: 'eth_requestAccounts' 
        });
        
        userAccount = accounts[0];
        console.log('✅ Wallet connected:', userAccount);
        
        // Switch to Polygon Amoy if needed
        await switchToPolygonAmoy();
        
        updateWalletUI();
        showNotification('Wallet connected to Polygon testnet!', 'success');
        
    } catch (error) {
        console.error('❌ Failed to connect wallet:', error);
        showNotification('Failed to connect wallet: ' + error.message, 'error');
    } finally {
        connectWalletBtn.disabled = false;
        updateConnectButtonText();
    }
}

// Switch to Polygon Amoy testnet
async function switchToPolygonAmoy() {
    try {
        await web3Provider.request({
            method: 'wallet_switchEthereumChain',
            params: [{ chainId: '0x13882' }], // 80002 in hex
        });
        console.log('✅ Switched to Polygon Amoy');
    } catch (switchError) {
        // Chain doesn't exist, add it
        if (switchError.code === 4902) {
            try {
                console.log('➕ Adding Polygon Amoy network...');
                await web3Provider.request({
                    method: 'wallet_addEthereumChain',
                    params: [
                        {
                            chainId: '0x13882',
                            chainName: 'Polygon Amoy Testnet',
                            nativeCurrency: {
                                name: 'POL',
                                symbol: 'POL',
                                decimals: 18,
                            },
                            rpcUrls: [POLYGON_RPC_URL],
                            blockExplorerUrls: ['https://amoy.polygonscan.com'],
                        },
                    ],
                });
                console.log('✅ Polygon Amoy network added');
            } catch (addError) {
                console.error('❌ Failed to add Polygon Amoy:', addError);
                throw addError;
            }
        } else {
            throw switchError;
        }
    }
}

// Check if we're on the correct network
async function checkNetwork() {
    try {
        const chainId = await web3Provider.request({ method: 'eth_chainId' });
        const currentChainId = parseInt(chainId, 16);
        
        if (currentChainId !== POLYGON_AMOY_CHAIN_ID) {
            console.log('⚠️ Wrong network detected:', currentChainId);
            showNotification('Please switch to Polygon Amoy testnet', 'error');
            return false;
        }
        
        return true;
    } catch (error) {
        console.error('❌ Failed to check network:', error);
        return false;
    }
}

// Lock analysis on-chain
async function lockOnChain() {
    if (!userAccount || !currentAnalysis) {
        showNotification('Connect wallet and generate analysis first!', 'error');
        return;
    }
    
    if (CONTRACT_ADDRESS === '0x...') {
        showNotification('Contract not deployed yet. Deploy the smart contract first!', 'error');
        return;
    }
    
    if (!(await checkNetwork())) {
        await switchToPolygonAmoy();
        return;
    }
    
    try {
        console.log('🔒 Starting on-chain lock process...');
        lockOnChainBtn.disabled = true;
        lockOnChainBtn.innerHTML = `
            <div class="loading-spinner w-4 h-4 mr-2"></div>
            Hashing Content...
        `;
        
        // Create SHA256 hash of the analysis
        const contentHash = CryptoJS.SHA256(currentAnalysis).toString();
        console.log('📝 Content hash created:', contentHash);
        
        // Determine category from analysis content
        const category = determineCategory(currentAnalysis);
        console.log('🏷️ Category determined:', category);
        
        lockOnChainBtn.innerHTML = `
            <div class="loading-spinner w-4 h-4 mr-2"></div>
            Signing Transaction...
        `;
        
        // Create contract instance for proper encoding
        const web3 = new Web3(web3Provider);
        const contract = new web3.eth.Contract(SIGNAL_REGISTRY_ABI, CONTRACT_ADDRESS);
        
        // Encode function call properly
        const encodedData = contract.methods.registerSignal(contentHash, category).encodeABI();
        
        // Estimate gas first
        let gasEstimate;
        try {
            gasEstimate = await web3.eth.estimateGas({
                from: userAccount,
                to: CONTRACT_ADDRESS,
                data: encodedData
            });
            console.log('⛽ Gas estimate (raw):', gasEstimate);
            console.log('⛽ Gas estimate type:', typeof gasEstimate);
        } catch (gasError) {
            console.error('❌ Gas estimation failed:', gasError);
            gasEstimate = BigInt(100000); // Fallback as BigInt
        }
        
        // Get current gas price
        let gasPrice;
        try {
            gasPrice = await web3.eth.getGasPrice();
            console.log('💰 Current gas price (raw):', gasPrice);
            console.log('💰 Gas price type:', typeof gasPrice);
        } catch (gasPriceError) {
            console.error('❌ Gas price fetch failed:', gasPriceError);
            gasPrice = BigInt(web3.utils.toWei('30', 'gwei')); // Fallback as BigInt
        }
        
        // Convert BigInt values to regular numbers for calculations
        const gasEstimateNum = Number(gasEstimate);
        const gasPriceNum = Number(gasPrice);
        
        console.log('⛽ Gas estimate (converted):', gasEstimateNum);
        console.log('💰 Gas price (converted):', gasPriceNum);
        
        // Calculate gas with 20% buffer
        const gasWithBuffer = Math.floor(gasEstimateNum * 1.2);
        console.log('⛽ Gas with 20% buffer:', gasWithBuffer);
        
        // Prepare transaction with converted values
        const transactionParameters = {
            to: CONTRACT_ADDRESS,
            from: userAccount,
            data: encodedData,
            gas: '0x' + gasWithBuffer.toString(16), // Convert to hex manually
            gasPrice: '0x' + gasPriceNum.toString(16) // Convert to hex manually
        };
        
        console.log('📋 Transaction params:', transactionParameters);
        
        // Send transaction
        const txHash = await web3Provider.request({
            method: 'eth_sendTransaction',
            params: [transactionParameters],
        });
        
        console.log('📤 Transaction sent:', txHash);
        
        lockOnChainBtn.innerHTML = `
            <div class="loading-spinner w-4 h-4 mr-2"></div>
            Confirming...
        `;
        
        // Wait for confirmation
        await waitForTransaction(txHash);
        
        // Success!
        lockOnChainBtn.innerHTML = `
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
            </svg>
            Locked On-Chain ✓
        `;
        lockOnChainBtn.classList.remove('from-green-500', 'to-green-600');
        lockOnChainBtn.classList.add('from-green-600', 'to-green-700', 'bg-gradient-to-r');
        
        // Show transaction success
        showTransactionSuccess(txHash, contentHash);
        
        // Animate $CREA rewards
        animateCreaRewards();
        
        console.log('✅ On-chain lock completed successfully!');
        
    } catch (error) {
        console.error('❌ Lock on-chain failed:', error);
        
        // Better error handling for common issues
        let errorMessage = 'Transaction failed: ';
        if (error.code === -32603) {
            errorMessage += 'Network error. Please check your connection and try again.';
        } else if (error.code === 4001) {
            errorMessage += 'Transaction cancelled by user.';
        } else if (error.message.includes('insufficient funds')) {
            errorMessage += 'Insufficient POL for gas fees. Get testnet POL from faucet.';
        } else if (error.message.includes('gas')) {
            errorMessage += 'Gas estimation failed. Try adjusting gas settings.';
        } else {
            errorMessage += error.message || 'Unknown error occurred.';
        }
        
        showNotification(errorMessage, 'error');
        resetLockButton();
    } finally {
        lockOnChainBtn.disabled = false;
    }
}

// Function encoding now handled by Web3.js contract methods

// Determine category from analysis content
function determineCategory(content) {
    const contentLower = content.toLowerCase();
    
    if (contentLower.includes('defi') || contentLower.includes('aave') || contentLower.includes('uni')) {
        return 'defi';
    } else if (contentLower.includes('ai') || contentLower.includes('artificial intelligence')) {
        return 'ai';
    } else if (contentLower.includes('gaming') || contentLower.includes('nft')) {
        return 'gaming';
    } else if (contentLower.includes('layer') || contentLower.includes('l1') || contentLower.includes('l2')) {
        return 'infrastructure';
    } else {
        return 'alpha';
    }
}

// Wait for transaction confirmation
async function waitForTransaction(txHash) {
    return new Promise((resolve, reject) => {
        const checkInterval = setInterval(async () => {
            try {
                const receipt = await web3Provider.request({
                    method: 'eth_getTransactionReceipt',
                    params: [txHash],
                });
                
                if (receipt) {
                    clearInterval(checkInterval);
                    if (receipt.status === '0x1') {
                        console.log('✅ Transaction confirmed:', receipt);
                        resolve(receipt);
                    } else {
                        console.error('❌ Transaction failed:', receipt);
                        reject(new Error('Transaction failed'));
                    }
                }
            } catch (error) {
                console.error('⚠️ Error checking transaction:', error);
            }
        }, 3000); // Check every 3 seconds
        
        // Timeout after 60 seconds
        setTimeout(() => {
            clearInterval(checkInterval);
            resolve(null); // Don't reject, just resolve with null
        }, 60000);
    });
}

// Reset lock button to original state
function resetLockButton() {
    lockOnChainBtn.innerHTML = `
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
        </svg>
        Lock-in On-Chain
    `;
    lockOnChainBtn.classList.remove('from-green-600', 'to-green-700');
    lockOnChainBtn.classList.add('from-green-500', 'to-green-600');
}

// Show transaction success with link to PolygonScan
function showTransactionSuccess(txHash, contentHash) {
    const polygonScanUrl = `https://amoy.polygonscan.com/tx/${txHash}`;
    
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 z-50 max-w-md';
    notification.innerHTML = `
        <div class="bg-green-600 text-white p-4 rounded-lg shadow-lg">
            <div class="flex items-center space-x-3 mb-3">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                <div>
                    <div class="font-semibold">Analysis Locked On-Chain!</div>
                    <div class="text-sm opacity-90">Proof stored on Polygon Amoy</div>
                </div>
            </div>
            <div class="text-xs bg-green-700 p-2 rounded mb-3 font-mono">
                Hash: ${contentHash.substring(0, 32)}...
            </div>
            <div class="flex space-x-2">
                <a href="${polygonScanUrl}" target="_blank" 
                   class="inline-flex items-center text-sm bg-green-700 px-3 py-1 rounded hover:bg-green-800 transition-colors">
                    View on PolygonScan
                    <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                    </svg>
                </a>
                <button onclick="this.parentElement.parentElement.remove()" 
                        class="text-sm bg-green-700 px-3 py-1 rounded hover:bg-green-800 transition-colors">
                    Close
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 15000);
}

// Animate $CREA rewards
function animateCreaRewards() {
    const currentValue = parseFloat(creaRewards.textContent.replace(/[^0-9.]/g, ''));
    const newValue = currentValue + 2.1;
    
    let step = 0;
    const steps = 30;
    const increment = 2.1 / steps;
    
    creaRewards.style.transition = 'all 0.1s ease';
    
    const animation = setInterval(() => {
        step++;
        const value = currentValue + (increment * step);
        creaRewards.textContent = `${value.toFixed(1)} $CREA`;
        creaRewards.style.transform = 'scale(1.1)';
        
        // Also update navbar counter if it exists
        const creaRewardsNav = document.getElementById('creaRewardsNav');
        if (creaRewardsNav) {
            creaRewardsNav.textContent = `${value.toFixed(1)} $CREA`;
        }
        
        setTimeout(() => {
            creaRewards.style.transform = 'scale(1)';
        }, 50);
        
        if (step >= steps) {
            clearInterval(animation);
            creaRewards.textContent = `${newValue.toFixed(1)} $CREA`;
            
            // Update navbar counter final value
            const creaRewardsNav = document.getElementById('creaRewardsNav');
            if (creaRewardsNav) {
                creaRewardsNav.textContent = `${newValue.toFixed(1)} $CREA`;
            }
            
            // Flash effect
            creaRewards.style.background = 'rgba(34, 197, 94, 0.2)';
            creaRewards.style.borderRadius = '4px';
            creaRewards.style.padding = '2px 4px';
            
            setTimeout(() => {
                creaRewards.style.background = '';
                creaRewards.style.borderRadius = '';
                creaRewards.style.padding = '';
            }, 1000);
        }
    }, 50);
}

// Update wallet UI when connected
function updateWalletUI() {
    if (userAccount) {
        connectWalletBtn.innerHTML = `
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
            </svg>
            <span class="hidden sm:inline">${userAccount.slice(0, 6)}...${userAccount.slice(-4)}</span>
            <span class="sm:hidden">✓</span>
            <svg class="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
            </svg>
        `;
        connectWalletBtn.classList.remove('from-blue-500', 'to-purple-500');
        connectWalletBtn.classList.add('from-green-500', 'to-green-600');
        connectWalletBtn.title = `Connected: ${userAccount}`;
        
        // Update dropdown address display
        walletAddressDisplay.textContent = `${userAccount.slice(0, 6)}...${userAccount.slice(-4)}`;
        
        // Show $CREA rewards counter when wallet connected
        const creaRewardsNavbar = document.getElementById('creaRewardsNavbar');
        if (creaRewardsNavbar) {
            creaRewardsNavbar.classList.remove('hidden');
        }
        
        // Show lock on-chain button in results section
        lockOnChainBtn.classList.remove('hidden');
        if (currentAnalysis) {
            lockOnChainBtn.disabled = false;
        }
        
        // Change button behavior to toggle dropdown
        connectWalletBtn.removeEventListener('click', connectWallet);
        connectWalletBtn.addEventListener('click', toggleWalletDropdown);
    }
}

function updateConnectButtonText() {
    if (!userAccount) {
        connectWalletBtn.innerHTML = `
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"></path>
            </svg>
            <span class="hidden sm:inline">Connect Wallet</span>
            <span class="sm:hidden">Wallet</span>
        `;
        connectWalletBtn.classList.remove('from-green-500', 'to-green-600');
        connectWalletBtn.classList.add('from-blue-500', 'to-purple-500');
        connectWalletBtn.title = 'Connect your Web3 wallet';
        
        // Reset to connect functionality
        connectWalletBtn.removeEventListener('click', toggleWalletDropdown);
        connectWalletBtn.addEventListener('click', connectWallet);
        
        // Hide dropdown
        walletDropdown.classList.add('hidden');
    }
}

// ============================================================================
// WALLET DROPDOWN FUNCTIONS
// ============================================================================

// Toggle wallet dropdown
function toggleWalletDropdown(e) {
    e.stopPropagation();
    walletDropdown.classList.toggle('hidden');
}

// Disconnect wallet
function disconnectWallet() {
    console.log('🔌 Disconnecting wallet...');
    userAccount = null;
    
    // Hide $CREA rewards counter when wallet disconnected
    const creaRewardsNavbar = document.getElementById('creaRewardsNavbar');
    if (creaRewardsNavbar) {
        creaRewardsNavbar.classList.add('hidden');
    }
    
    lockOnChainBtn.classList.add('hidden');
    lockOnChainBtn.disabled = true;
    updateConnectButtonText();
    showNotification('Wallet disconnected', 'info');
}

// Copy wallet address
async function copyWalletAddress() {
    if (!userAccount) return;
    
    try {
        await navigator.clipboard.writeText(userAccount);
        showNotification('Wallet address copied!', 'success');
    } catch (error) {
        showNotification('Failed to copy address', 'error');
    }
}

// View wallet on explorer
function viewWalletOnExplorer() {
    if (!userAccount) return;
    
    const explorerUrl = `https://amoy.polygonscan.com/address/${userAccount}`;
    window.open(explorerUrl, '_blank');
    showNotification('Opening PolygonScan...', 'info');
}

// Close dropdown when clicking outside
function handleClickOutside(e) {
    if (!walletDropdown.contains(e.target) && !connectWalletBtn.contains(e.target)) {
        walletDropdown.classList.add('hidden');
    }
}

// Handle account changes
function handleAccountsChanged(accounts) {
    if (accounts.length === 0) {
        console.log('🔌 Wallet disconnected');
        userAccount = null;
        lockOnChainBtn.classList.add('hidden');
        lockOnChainBtn.disabled = true;
        walletDropdown.classList.add('hidden');
        updateConnectButtonText();
        showNotification('Wallet disconnected', 'info');
    } else {
        userAccount = accounts[0];
        console.log('🔄 Account changed to:', userAccount);
        walletDropdown.classList.add('hidden'); // Close dropdown on account change
        updateWalletUI();
    }
}

// Handle chain changes
function handleChainChanged(chainId) {
    console.log('🔄 Chain changed to:', chainId);
    const currentChainId = parseInt(chainId, 16);
    
    if (currentChainId !== POLYGON_AMOY_CHAIN_ID) {
        showNotification('Please switch to Polygon Amoy testnet', 'error');
        lockOnChainBtn.disabled = true;
    } else {
        showNotification('Connected to Polygon Amoy testnet', 'success');
        if (currentAnalysis && userAccount) {
            lockOnChainBtn.disabled = false;
        }
    }
}

// List on marketplace (REAL implementation - calls backend API)
async function listOnMarketplace() {
    if (!currentAnalysis) {
        showNotification('Generate analysis first!', 'error');
        return;
    }
    
    listMarketplaceBtn.disabled = true;
    const originalHTML = listMarketplaceBtn.innerHTML;
    listMarketplaceBtn.innerHTML = `
        <div class="loading-spinner w-4 h-4 mr-2"></div>
        Listing...
    `;
    
    try {
        // Extract title from analysis (first line or use default)
        const lines = currentAnalysis.split('\n').filter(line => line.trim());
        const title = lines.find(line => line.includes('#')) 
            ? lines.find(line => line.includes('#')).replace(/#+\s*/, '').trim()
            : 'Alpha Analysis';
        
        // Call backend API
        const response = await fetch('/api/marketplace/list', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                content: currentAnalysis,
                title: title
            })
        });

        const data = await response.json();
        
        if (data.success) {
            // Success state
            listMarketplaceBtn.innerHTML = `
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                Listed! ✓
            `;
            listMarketplaceBtn.classList.remove('from-purple-500', 'to-pink-500');
            listMarketplaceBtn.classList.add('from-green-500', 'to-green-600');
            
            // Show success notification with link
            const notification = document.createElement('div');
            notification.className = 'fixed top-4 right-4 bg-green-600 text-white px-6 py-4 rounded-lg shadow-lg z-50 max-w-sm';
            notification.innerHTML = `
                <div class="flex items-start space-x-3">
                    <i class="fas fa-check-circle text-xl"></i>
                    <div>
                        <div class="font-bold">Listed on Marketplace!</div>
                        <div class="text-sm opacity-90 mt-1">Public URL created successfully</div>
                        <a href="${data.public_url}" target="_blank" 
                           class="text-sm underline hover:no-underline mt-2 block">
                            View Public Listing →
                        </a>
                    </div>
                </div>
            `;
            document.body.appendChild(notification);
            
            // Auto-remove notification
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.style.opacity = '0';
                    notification.style.transform = 'translateX(100%)';
                    setTimeout(() => notification.remove(), 300);
                }
            }, 5000);
            
            // Animate $CREA rewards
            animateCreaRewards();
            
        } else {
            throw new Error(data.error || 'Failed to create listing');
        }
        
    } catch (error) {
        console.error('Marketplace listing error:', error);
        showNotification('Failed to list on marketplace. Please try again.', 'error');
        
        // Reset button immediately on error
        listMarketplaceBtn.disabled = false;
        listMarketplaceBtn.innerHTML = originalHTML;
        return;
    }
    
    // Reset button after success (3 seconds)
    setTimeout(() => {
        listMarketplaceBtn.disabled = false;
        listMarketplaceBtn.innerHTML = originalHTML;
        listMarketplaceBtn.classList.remove('from-green-500', 'to-green-600');
        listMarketplaceBtn.classList.add('from-purple-500', 'to-pink-500');
    }, 3000);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

// Web3 event listeners
connectWalletBtn?.addEventListener('click', connectWallet);
lockOnChainBtn?.addEventListener('click', lockOnChain);
listMarketplaceBtn?.addEventListener('click', listOnMarketplace);

// Wallet dropdown event listeners
disconnectWalletBtn?.addEventListener('click', disconnectWallet);
copyAddressBtn?.addEventListener('click', copyWalletAddress);
viewOnExplorerBtn?.addEventListener('click', viewWalletOnExplorer);

// Close dropdown when clicking outside
document.addEventListener('click', handleClickOutside);

// Sample queries for demo
const sampleQueries = [
    "AAVE UNI COMP",
    "Top 3 AI tokens",
    "Trending DeFi", 
    "BTC ETH SOL prices"
];

// Initialize on page load
window.addEventListener('load', () => {
    console.log('🚀 Initializing Sovereign Agent #001...');
    
    // Initialize Web3
    initWeb3();
    
    // Set sample query
    const randomQuery = sampleQueries[Math.floor(Math.random() * sampleQueries.length)];
    queryInput.placeholder = randomQuery;
    
    console.log('✅ Sovereign Agent #001 ready!');
});

// ============================================================================
// EXISTING FUNCTIONALITY (AI Generation, Copy, etc.)
// ============================================================================

// Pre-filled prompt template functionality
promptTemplates.forEach(template => {
    template.addEventListener('click', () => {
        const prompt = template.getAttribute('data-prompt');
        queryInput.value = prompt;
        queryInput.focus();
        // Auto-generate for convenience
        setTimeout(() => {
            generateBtn.click();
        }, 500);
    });
});

// Generate alpha analysis
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

    // Hide Quick Start Templates during analysis for cleaner UX
    quickStartTemplates.style.display = 'none';

    // START AGENTIC WORKFLOW VISUALIZATION (Critical for demo "wow" factor)
    await showAgenticWorkflow();
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const data = await response.json();
        
        if (data.success) {
            showResults(data);
            showNotification('Analysis generated successfully!', 'success');
            
            // Enable lock on-chain if wallet connected
            if (userAccount && CONTRACT_ADDRESS !== '0x...') {
                lockOnChainBtn.disabled = false;
            }
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
        
        // Show Quick Start Templates again after analysis completes
        quickStartTemplates.style.display = 'block';
    }
});

// Copy functionality
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

// ============================================================================
// UI STATE MANAGEMENT
// ============================================================================

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

// ============================================================================
// AGENTIC WORKFLOW VISUALIZATION (Critical for demo "wow" factor)
// ============================================================================

async function showAgenticWorkflow() {
    const stages = [
        {
            icon: 'fas fa-brain',
            text: 'Planning...',
            description: 'Planning Agent creating research strategy',
            color: 'text-blue-400',
            bgColor: 'border-blue-400',
            duration: 2000
        },
        {
            icon: 'fas fa-search',
            text: 'Scouting...',
            description: 'Scout Agent gathering real-time data',
            color: 'text-yellow-400',
            bgColor: 'border-yellow-400',
            duration: 2500
        },
        {
            icon: 'fas fa-chart-line',
            text: 'Analyzing...',
            description: 'Analyst Agent synthesizing investment thesis',
            color: 'text-green-400',
            bgColor: 'border-green-400',
            duration: 3000
        },
        {
            icon: 'fas fa-shield-alt',
            text: 'Risk Checking...',
            description: 'Risk Analyst Agent quality control & validation',
            color: 'text-red-400',
            bgColor: 'border-red-400',
            duration: 2500
        },
        {
            icon: 'fas fa-broadcast-tower',
            text: 'Finalizing...',
            description: 'Comms Agent formatting for publication',
            color: 'text-purple-400',
            bgColor: 'border-purple-400',
            duration: 2000
        },
        {
            icon: 'fas fa-check-circle',
            text: 'Complete!',
            description: 'Alpha Squad analysis ready',
            color: 'text-green-400',
            bgColor: 'border-green-400',
            duration: 1000
        }
    ];

    // Show the loading state container
    hideAllStates();
    loadingState.classList.remove('hidden');
    
    // Get the loading content container
    const loadingContent = loadingState.querySelector('.loading-content') || loadingState;

    for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        const progressPercent = ((i + 1) / stages.length) * 100;
        
        // Create stage visualization
        loadingContent.innerHTML = `
            <div class="text-center space-y-6">
                <!-- Agent Icon -->
                <div class="flex justify-center">
                    <div class="w-20 h-20 rounded-full bg-gray-800 flex items-center justify-center border-3 ${stage.bgColor} relative">
                        <i class="${stage.icon} text-3xl ${stage.color} animate-pulse"></i>
                        <div class="absolute -inset-1 rounded-full border-2 ${stage.bgColor} animate-ping opacity-25"></div>
                    </div>
                </div>
                
                <!-- Stage Info -->
                <div class="space-y-3">
                    <h3 class="text-2xl font-bold ${stage.color}">${stage.text}</h3>
                    <p class="text-gray-300 text-lg">${stage.description}</p>
                </div>
                
                <!-- Progress Bar -->
                <div class="space-y-2">
                    <div class="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                        <div class="bg-gradient-to-r from-blue-500 via-purple-500 to-green-500 h-3 rounded-full transition-all duration-1000 ease-out transform origin-left" 
                             style="width: ${progressPercent}%; animation: shimmer 2s infinite;"></div>
                    </div>
                    <div class="text-sm text-gray-400">
                        Alpha Squad Agent ${i + 1} of ${stages.length - 1} • Autonomous Analysis System
                    </div>
                </div>
                
                <!-- Real-time Status -->
                <div class="bg-gray-800 rounded-lg p-4 border border-gray-600">
                    <div class="flex items-center justify-center space-x-2 text-sm text-gray-300">
                        <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                        <span>Autonomous Creator Protocol • Live Analysis</span>
                    </div>
                </div>
            </div>
        `;
        
        // Wait for stage duration with realistic timing
        await new Promise(resolve => setTimeout(resolve, stage.duration));
    }
    
    // Add final transition effect
    loadingContent.innerHTML = `
        <div class="text-center space-y-4">
            <div class="flex justify-center">
                <div class="w-16 h-16 rounded-full bg-green-600 flex items-center justify-center animate-bounce">
                    <i class="fas fa-check text-2xl text-white"></i>
                </div>
            </div>
            <h3 class="text-xl font-bold text-green-400">Analysis Complete!</h3>
            <p class="text-gray-300">Displaying results...</p>
        </div>
    `;
    
    await new Promise(resolve => setTimeout(resolve, 800));
}

// Add required CSS animations
const agenticStyles = document.createElement('style');
agenticStyles.textContent = `
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .animate-ping {
        animation: ping 1s cubic-bezier(0, 0, 0.2, 1) infinite;
    }
    
    @keyframes ping {
        75%, 100% {
            transform: scale(2);
            opacity: 0;
        }
    }
    
    .border-3 {
        border-width: 3px;
    }
`;
document.head.appendChild(agenticStyles);

// ============================================================================
// SOCIAL MEDIA SHARING FUNCTIONALITY
// ============================================================================

// Add event listeners for social sharing
document.addEventListener('DOMContentLoaded', function() {
    const shareBtn = document.getElementById('shareBtn');
    const shareDropdown = document.getElementById('shareDropdown');
    const shareTwitter = document.getElementById('shareTwitter');
    const shareLinkedIn = document.getElementById('shareLinkedIn');
    const shareReddit = document.getElementById('shareReddit');
    const shareTelegram = document.getElementById('shareTelegram');
    const copyLink = document.getElementById('copyLink');

    // Toggle share dropdown
    if (shareBtn && shareDropdown) {
        shareBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            shareDropdown.classList.toggle('hidden');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function() {
            shareDropdown.classList.add('hidden');
        });

        // Prevent dropdown from closing when clicking inside
        shareDropdown.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    }

    // Social media sharing functions
    function shareToSocial(platform) {
        if (!currentAnalysis) {
            showNotification('Generate analysis first!', 'error');
            return;
        }

        const title = 'Fresh Alpha Insights from Sovereign Agent #001';
        const description = 'AI-generated crypto analysis from the Alpha Squad';
        const url = window.location.href;
        const hashtags = 'crypto,alpha,AI,DeFi,blockchain';

        let shareUrl = '';
        const analysisPreview = currentAnalysis.substring(0, 200) + '...';

        switch (platform) {
            case 'twitter':
                shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(`${title}\n\n${analysisPreview}\n\n#${hashtags.replace(/,/g, ' #')}`)}`;
                break;
            case 'linkedin':
                shareUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}&summary=${encodeURIComponent(description)}`;
                break;
            case 'reddit':
                shareUrl = `https://reddit.com/submit?url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`;
                break;
            case 'telegram':
                shareUrl = `https://t.me/share/url?url=${encodeURIComponent(url)}&text=${encodeURIComponent(title + '\n\n' + description)}`;
                break;
        }

        if (shareUrl) {
            window.open(shareUrl, '_blank', 'width=600,height=400');
            shareDropdown.classList.add('hidden');
            showNotification(`Shared to ${platform.charAt(0).toUpperCase() + platform.slice(1)}!`, 'success');
        }
    }

    // Copy link functionality
    function copyShareLink() {
        if (!currentAnalysis) {
            showNotification('Generate analysis first!', 'error');
            return;
        }

        const shareText = `🎯 Fresh Alpha Insights from Sovereign Agent #001\n\n${currentAnalysis.substring(0, 200)}...\n\nGenerated by AI Alpha Squad • ${window.location.href}`;
        
        navigator.clipboard.writeText(shareText).then(() => {
            showNotification('Analysis link copied to clipboard!', 'success');
            shareDropdown.classList.add('hidden');
        }).catch(() => {
            showNotification('Failed to copy link', 'error');
        });
    }

    // Attach event listeners
    if (shareTwitter) shareTwitter.addEventListener('click', () => shareToSocial('twitter'));
    if (shareLinkedIn) shareLinkedIn.addEventListener('click', () => shareToSocial('linkedin'));
    if (shareReddit) shareReddit.addEventListener('click', () => shareToSocial('reddit'));
    if (shareTelegram) shareTelegram.addEventListener('click', () => shareToSocial('telegram'));
    if (copyLink) copyLink.addEventListener('click', copyShareLink);
});