const { ethers } = require('hardhat');

async function main() {
    console.log("🚀 Deploying SignalRegistry to Polygon Amoy testnet...");
    console.log("Network:", hre.network.name);
    
    // Get the deployer account
    const [deployer] = await ethers.getSigners();
    console.log("Deploying with account:", deployer.address);
    
    // Check balance
    const balance = await ethers.provider.getBalance(deployer.address);
    console.log("Account balance:", ethers.formatEther(balance), "POL");
    
    if (balance < ethers.parseEther("0.01")) {
        console.log("⚠️  Warning: Low balance. Get testnet POL from https://faucet.polygon.technology/");
    }
    
    // Deploy the contract
    console.log("\n📝 Compiling and deploying SignalRegistry...");
    const SignalRegistry = await ethers.getContractFactory("SignalRegistry");
    
    console.log("Estimated gas for deployment...");
    
    const signalRegistry = await SignalRegistry.deploy();
    
    console.log("🔄 Transaction sent, waiting for confirmation...");
    console.log("Transaction hash:", signalRegistry.deploymentTransaction().hash);
    
    await signalRegistry.waitForDeployment();
    
    console.log("\n✅ SignalRegistry deployed successfully!");
    console.log("📍 Contract address:", await signalRegistry.getAddress());
    console.log("🔗 View on PolygonScan:", `https://amoy.polygonscan.com/address/${await signalRegistry.getAddress()}`);
    console.log("🔗 Transaction:", `https://amoy.polygonscan.com/tx/${signalRegistry.deploymentTransaction().hash}`);
    
    // Test the deployment with a sample call
    console.log("\n🧪 Testing deployment...");
    try {
        const totalSignals = await signalRegistry.totalSignals();
        console.log("✅ Contract is responsive. Total signals:", totalSignals.toString());
        
        // Test signal count for deployer
        const deployerSignalCount = await signalRegistry.getSignalCount(deployer.address);
        console.log("✅ Signal count for deployer:", deployerSignalCount.toString());
        
    } catch (error) {
        console.log("❌ Contract test failed:", error.message);
    }
    
    // Verify on PolygonScan (only on testnets/mainnet)
    if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
        console.log("\n⏳ Waiting for block confirmations before verification...");
        await signalRegistry.deploymentTransaction().wait(6);
        
        try {
            console.log("🔍 Verifying contract on PolygonScan...");
            await hre.run("verify:verify", {
                address: await signalRegistry.getAddress(),
                constructorArguments: [],
            });
            console.log("✅ Contract verified successfully!");
        } catch (error) {
            console.log("❌ Verification failed:", error.message);
            console.log("You can manually verify at https://amoy.polygonscan.com/verifyContract");
        }
    }
    
    // Output summary for frontend integration
    console.log("\n📋 INTEGRATION SUMMARY:");
    console.log("=".repeat(50));
    console.log("Network: Polygon Amoy (Chain ID: 80002)");
    console.log("Contract Address:", await signalRegistry.getAddress());
    console.log("RPC URL: https://rpc-amoy.polygon.technology");
    console.log("Explorer: https://amoy.polygonscan.com");
    console.log("Faucet: https://faucet.polygon.technology/");
    console.log("=".repeat(50));
    console.log("\n💡 Next steps:");
    console.log("1. Add contract address to your .env file");
    console.log("2. Update CONTRACT_ADDRESS in dashboard.js");
    console.log("3. Get testnet POL from faucet for testing");
    console.log("4. Test the Lock-in On-Chain feature!");
}

main()
    .then(() => {
        console.log("\n🎉 Deployment completed successfully!");
        process.exit(0);
    })
    .catch((error) => {
        console.error("\n💥 Deployment failed:", error);
        process.exit(1);
    });
