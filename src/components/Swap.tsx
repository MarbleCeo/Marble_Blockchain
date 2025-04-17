import React, { useState, useEffect } from 'react';
import { TOKENS, Token } from '../utils/tokens';
import { fetchTrade, fetchBridge, fetchValidateBlock, fetchAnalyzeBlock } from '../utils/connection';
import './Swap.css';

interface SwapProps {
  walletAddress?: string;
}

const Swap: React.FC<SwapProps> = ({ walletAddress }) => {
  const [fromToken, setFromToken] = useState<Token>(TOKENS.SOL);
  const [toToken, setToToken] = useState<Token>(TOKENS.USDC);
  const [amount, setAmount] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [slippage, setSlippage] = useState<number>(0.5);
  const [bridgeAmount, setBridgeAmount] = useState<string>('');
  const [isBridging, setIsBridging] = useState<boolean>(false);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  const handleSwap = async () => {
    if (!walletAddress || !amount || parseFloat(amount) <= 0) return;
    
    setIsLoading(true);
    try {
      // Use a random recipient for demo purposes
      const recipient = "DEMO_RECIPIENT_ADDRESS";
      await fetchTrade(walletAddress, recipient, parseFloat(amount), fromToken.symbol);
      alert(`Successfully queued trade of ${amount} ${fromToken.symbol}`);
    } catch (error) {
      console.error('Swap failed:', error);
      alert('Swap failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBridge = async () => {
    if (!walletAddress || !bridgeAmount || parseFloat(bridgeAmount) <= 0) return;
    
    setIsBridging(true);
    try {
      await fetchBridge(fromToken.mint, parseFloat(bridgeAmount));
      alert(`Successfully bridged ${bridgeAmount} ${fromToken.symbol} to Marble`);
    } catch (error) {
      console.error('Bridge failed:', error);
      alert('Bridge failed. Please try again.');
    } finally {
      setIsBridging(false);
    }
  };

  const handleValidateBlock = async () => {
    // Using mock data for demonstration
    const mockBlockId = "block_123456";
    const mockCoins = 10.5;
    
    setIsValidating(true);
    try {
      const result = await fetchValidateBlock(mockBlockId, mockCoins);
      alert(`Block ${mockBlockId} validated successfully with power: ${result.power}`);
    } catch (error) {
      console.error('Block validation failed:', error);
      alert('Block validation failed. Please try again.');
    } finally {
      setIsValidating(false);
    }
  };

  const handleAnalyzeBlock = async () => {
    // Using mock data for demonstration
    const mockBlockId = "block_123456";
    
    setIsAnalyzing(true);
    try {
      const result = await fetchAnalyzeBlock(mockBlockId);
      alert(`Block ${mockBlockId} analysis complete: ${JSON.stringify(result)}`);
    } catch (error) {
      console.error('Block analysis failed:', error);
      alert('Block analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTokenSwitch = () => {
    const temp = fromToken;
    setFromToken(toToken);
    setToToken(temp);
  };

  return (
    <div className="swap-container card">
      <h2>Marble DEX Swap</h2>
      <p className="swap-description">Swap tokens instantly with Marble DEX</p>
      
      <div className="token-input-container">
        <label>From</label>
        <div className="token-input">
          <select 
            value={fromToken.symbol}
            onChange={(e) => setFromToken(TOKENS[e.target.value])}
          >
            {Object.values(TOKENS).map((token) => (
              <option key={token.mint} value={token.symbol}>
                {token.symbol}
              </option>
            ))}
          </select>
          <input
            type="number"
            placeholder="0.00"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            min="0"
          />
        </div>
      </div>
      
      <button className="switch-button" onClick={handleTokenSwitch}>
        ↕️
      </button>
      
      <div className="token-input-container">
        <label>To</label>
        <div className="token-input">
          <select
            value={toToken.symbol}
            onChange={(e) => setToToken(TOKENS[e.target.value])}
          >
            {Object.values(TOKENS).map((token) => (
              <option key={token.mint} value={token.symbol}>
                {token.symbol}
              </option>
            ))}
          </select>
          <input
            type="number"
            placeholder="0.00"
            value={amount} // In a real app, this would show estimated received amount
            disabled
          />
        </div>
      </div>
      
      <div className="settings-row">
        <label>Slippage Tolerance</label>
        <div className="slippage-buttons">
          <button 
            className={slippage === 0.1 ? "active" : ""} 
            onClick={() => setSlippage(0.1)}
          >
            0.1%
          </button>
          <button 
            className={slippage === 0.5 ? "active" : ""} 
            onClick={() => setSlippage(0.5)}
          >
            0.5%
          </button>
          <button 
            className={slippage === 1.0 ? "active" : ""} 
            onClick={() => setSlippage(1.0)}
          >
            1.0%
          </button>
        </div>
      </div>
      
      <button 
        className="swap-button" 
        onClick={handleSwap}
        disabled={isLoading || !walletAddress || !amount || parseFloat(amount) <= 0}
      >
        {isLoading ? "Processing..." : "Swap"}
      </button>

      {/* New Bridge to Marble section */}
      <div className="bridge-container">
        <h3>Bridge to Marble</h3>
        <div className="token-input">
          <select 
            value={fromToken.symbol}
            onChange={(e) => setFromToken(TOKENS[e.target.value])}
          >
            {Object.values(TOKENS).map((token) => (
              <option key={token.mint} value={token.symbol}>
                {token.symbol}
              </option>
            ))}
          </select>
          <input
            type="number"
            placeholder="Amount to bridge"
            value={bridgeAmount}
            onChange={(e) => setBridgeAmount(e.target.value)}
            min="0"
          />
        </div>
        <button 
          className="bridge-button" 
          onClick={handleBridge}
          disabled={isBridging || !walletAddress || !bridgeAmount || parseFloat(bridgeAmount) <= 0}
        >
          {isBridging ? "Bridging..." : "Bridge to Marble"}
        </button>
      </div>

      {/* Block validation and analysis section */}
      <div className="blockchain-tools-container">
        <h3>Blockchain Tools</h3>
        <div className="blockchain-tools-buttons">
          <button 
            className="validate-button" 
            onClick={handleValidateBlock}
            disabled={isValidating}
          >
            {isValidating ? "Validating..." : "Validate Block"}
          </button>
          <button 
            className="analyze-button" 
            onClick={handleAnalyzeBlock}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? "Analyzing..." : "Analyze Block"}
          </button>
        </div>
      </div>

      <div className="info-text">
        <p>Powered by Marble DEX - Fast, Secure, Decentralized</p>
      </div>
    </div>
  );
};

export default Swap;

