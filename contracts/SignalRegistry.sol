// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title SignalRegistry
 * @dev Simple contract to register content hashes on-chain for Alpha Co-Pilot demo
 * Deployed on Polygon Testnet (Mumbai) for fast, cheap transactions
 */
contract SignalRegistry {
    struct Signal {
        string contentHash;
        uint256 timestamp;
        address creator;
        string category; // e.g., "alpha", "defi", "ai-tokens"
    }
    
    // Mapping from creator address to their content hashes
    mapping(address => string[]) public signalHashes;
    
    // Mapping from content hash to signal details
    mapping(string => Signal) public signals;
    
    // Total number of signals registered
    uint256 public totalSignals;
    
    // Events
    event SignalRegistered(
        address indexed creator, 
        string indexed contentHash, 
        uint256 timestamp,
        string category
    );
    
    /**
     * @dev Register a new alpha signal on-chain
     * @param contentHash SHA256 hash of the generated alpha content
     * @param category Category of the content (optional)
     */
    function registerSignal(string memory contentHash, string memory category) public {
        require(bytes(contentHash).length > 0, "Content hash cannot be empty");
        require(bytes(signals[contentHash].contentHash).length == 0, "Signal already exists");
        
        // Store the signal
        signalHashes[msg.sender].push(contentHash);
        signals[contentHash] = Signal({
            contentHash: contentHash,
            timestamp: block.timestamp,
            creator: msg.sender,
            category: category
        });
        
        totalSignals++;
        
        emit SignalRegistered(msg.sender, contentHash, block.timestamp, category);
    }
    
    /**
     * @dev Get the number of signals created by an address
     */
    function getSignalCount(address creator) public view returns (uint256) {
        return signalHashes[creator].length;
    }
    
    /**
     * @dev Get signal details by content hash
     */
    function getSignalByHash(string memory contentHash) public view returns (Signal memory) {
        return signals[contentHash];
    }
    
    /**
     * @dev Get all signal hashes for a creator
     */
    function getCreatorSignals(address creator) public view returns (string[] memory) {
        return signalHashes[creator];
    }
    
    /**
     * @dev Verify if a content hash exists and was created by the specified address
     */
    function verifySignal(string memory contentHash, address creator) public view returns (bool) {
        Signal memory signal = signals[contentHash];
        return signal.creator == creator && bytes(signal.contentHash).length > 0;
    }
}
