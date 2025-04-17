import React, { useState, useEffect } from 'react';
import { fetchDexConfig, fetchValidateBlock, fetchAnalyzeBlock } from '../utils/connection';

// Define types
interface Token {
  symbol: string;
  name: string;
  balance: number;
  address: string;
}

interface SwapState {
  fromToken: Token;
  toToken: Token;
  fromAmount: string;
  toAmount: string;
  slippage: number;
}

const Swap: React.FC = () => {
  // State for swap functionality
  const [swapState, setSwapState] = useState<SwapState>({
    fromToken: { symbol: 'MARBLE', name: 'Marble Token', balance: 100, address: '0x123' },
    toToken: { symbol: 'SOL', name: 'Solana', balance: 10, address: '0x456' },
    fromAmount: '',
    toAmount: '',
    slippage: 0.5,
  });

  // Tokens available for swapping
  const [availableTokens, setAvailableTokens] = useState<Token[]>([
    { symbol: 'MARBLE', name: 'Marble Token', balance: 100, address: '0x123' },
    { symbol: 'SOL', name: 'Solana', balance: 10, address: '0x456' },
    { symbol: 'USDC', name: 'USD Coin', balance: 500, address: '0x789' },
  ]);

  // State for block validation and analysis
  const [blockId, setBlockId] = useState<string>('');
  const [validationCoins, setValidationCoins] = useState<string>('10');
  const [validationResult, setValidationResult] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  
  // Loading and error states
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Exchange rate calculation (simplified)
  const calculateExchangeRate = (from: string, to: string): number => {
    const rates: Record<string, Record<string, number>> = {
      'MARBLE': { 'SOL': 0.05, 'USDC': 2.5 },
      'SOL': { 'MARBLE': 20, 'USDC': 50 },
      'USDC': { 'MARBLE': 0.4, 'SOL': 0.02 },
    };
    return rates[from]?.[to] || 1;
  };

  // Handle from amount change
  const handleFromAmountChange = (value: string) => {
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
      const numValue = parseFloat(value) || 0;
      const rate = calculateExchangeRate(swapState.fromToken.symbol, swapState.toToken.symbol);
      
      setSwapState({
        ...swapState,
        fromAmount: value,
        toAmount: numValue > 0 ? (numValue * rate).toFixed(6) : '',
      });
    }
  };

  // Handle to amount change
  const handleToAmountChange = (value: string) => {
    if (value === '' || /^\d*\.?\d*$/.test(value)) {
      const numValue = parseFloat(value) || 0;
      const rate = calculateExchangeRate(swapState.toToken.symbol, swapState.fromToken.symbol);
      
      setSwapState({
        ...swapState,
        toAmount: value,
        fromAmount: numValue > 0 ? (numValue * rate).toFixed(6) : '',
      });
    }
  };

  // Token selection handlers
  const handleFromTokenChange = (symbol: string) => {
    const token = availableTokens.find(t => t.symbol === symbol);
    if (token) {
      setSwapState({
        ...swapState,
        fromToken: token,
        fromAmount: '',
        toAmount: '',
      });
    }
  };

  const handleToTokenChange = (symbol: string) => {
    const token = availableTokens.find(t => t.symbol === symbol);
    if (token) {
      setSwapState({
        ...swapState,
        toToken: token,
        fromAmount: '',
        toAmount: '',
      });
    }
  };

  // Swap execution
  const executeSwap = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Mock API call for swap
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Reset form after successful swap
      setSwapState({
        ...swapState,
        fromAmount: '',
        toAmount: '',
      });
      
      setIsLoading(false);
      
      // Update token balances (simplified)
      const fromAmount = parseFloat(swapState.fromAmount);
      const toAmount = parseFloat(swapState.toAmount);
      
      setAvailableTokens(prev => prev.map(token => {
        if (token.symbol === swapState.fromToken.symbol) {
          return { ...token, balance: token.balance - fromAmount };
        }
        if (token.symbol === swapState.toToken.symbol) {
          return { ...token, balance: token.balance + toAmount };
        }
        return token;
      }));
      
    } catch (err) {
      setIsLoading(false);
      setError('Swap failed. Please try again.');
      console.error('Swap error:', err);
    }
  };

  // Bridge to Marble (mock)
  const bridgeToMarble = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Mock API call for bridge
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setIsLoading(false);
      alert('Bridge transaction submitted successfully!');
      
    } catch (err) {
      setIsLoading(false);
      setError('Bridge failed. Please try again.');
      console.error('Bridge error:', err);
    }
  };

  // Block validation function
  const validateBlock = async () => {
    if (!blockId) {
      setError('Please enter a block ID');
      return;
    }
    
    try {
      setIsLoading(true);
      setError(null);
      setValidationResult(null);
      
      const response = await fetchValidateBlock(blockId, validationCoins);
      
      setValidationResult(response.valid 
        ? `Block ${blockId} is valid! Validation complete.` 
        : `Block ${blockId} validation failed: ${response.reason}`);
      
      setIsLoading(false);
    } catch (err) {
      setIsLoading(false);
      setError('Block validation failed. Please try again.');
      console.error('Validation error:', err);
    }
  };

  // Block analysis function
  const analyzeBlock = async () => {
    if (!blockId) {
      setError('Please enter a block ID');
      return;
    }
    
    try {
      setIsLoading(true);
      setError(null);
      setAnalysisResult(null);
      
      const response = await fetchAnalyzeBlock(blockId);
      
      setAnalysisResult(
        `Block ${blockId} Analysis:\n` +
        `Transactions: ${response.transactions_count}\n` +
        `Volume: ${response.volume} MARBLE\n` +
        `AI Analysis: ${response.ai_analysis}`
      );
      
      setIsLoading(false);
    } catch (err) {
      setIsLoading(false);
      setError('Block analysis failed. Please try again.');
      console.error('Analysis error:', err);
    }
  };

  return (
    <div className="swap-container">
      <h2>Marble DEX Swap</h2>
      
      {/* Swap Interface */}
      <div className="swap-box">
        <div className="swap-input-container">
          <div className="swap-input-header">
            <span>From</span>
            <span>Balance: {swapState.fromToken.balance} {swapState.fromToken.symbol}</span>
          </div>
          <div className="swap-input-group">
            <input
              type="text"
              className="swap-input"
              value={swapState.fromAmount}
              onChange={(e) => handleFromAmountChange(e.target.value)}
              placeholder="0.0"
              disabled={isLoading}
            />
            <select
              className="token-select"
              value={swapState.fromToken.symbol}
              onChange={(e) => handleFromTokenChange(e.target.value)}
              disabled={isLoading}
            >
              {availableTokens.map(token => (
                <option key={token.symbol} value={token.symbol}>
                  {token.symbol}
                </option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="swap-arrow">↓</div>
        
        <div className="swap-input-container">
          <div className="swap-input-header">
            <span>To</span>
            <span>Balance: {swapState.toToken.balance} {swapState.toToken.symbol}</span>
          </div>
          <div className="swap-input-group">
            <input
              type="text"
              className="swap-input"
              value={swapState.toAmount}
              onChange={(e) => handleToAmountChange(e.target.value)}
              placeholder="0.0"
              disabled={isLoading}
            />
            <select
              className="token-select"
              value={swapState.toToken.symbol}
              onChange={(e) => handleToTokenChange(e.target.value)}
              disabled={isLoading}
            >
              {availableTokens
                .filter(token => token.symbol !== swapState.fromToken.symbol)
                .map(token => (
                  <option key={token.symbol} value={token.symbol}>
                    {token.symbol}
                  </option>
                ))}
            </select>
          </div>
        </div>
        
        {parseFloat(swapState.fromAmount) > 0 && (
          <div className="swap-details">
            <div className="swap-detail-item">
              <span>Rate:</span>
              <span>1 {swapState.fromToken.symbol} = {calculateExchangeRate(swapState.fromToken.symbol, swapState.toToken.symbol)} {swapState.toToken.symbol}</span>
            </div>
            <div className="swap-detail-item">
              <span>Slippage:</span>
              <span>{swapState.slippage}%</span>
            </div>
          </div>
        )}
        
        <button
          className="swap-button"
          onClick={executeSwap}
          disabled={isLoading || !parseFloat(swapState.fromAmount) || parseFloat(swapState.fromAmount) > swapState.fromToken.balance}
        >
          {isLoading ? 'Processing...' : 'Swap'}
        </button>
        
        <button
          className="bridge-button"
          onClick={bridgeToMarble}
          disabled={isLoading}
        >
          Bridge to Marble
        </button>
      </div>
      
      {/* Block Operations */}
      <div className="block-operations">
        <h3>Block Operations</h3>
        
        <div className="block-id-input">
          <input
            type="text"
            placeholder="Enter Block ID"
            value={blockId}
            onChange={(e) => setBlockId(e.target.value)}
            disabled={isLoading}
          />
        </div>
        
        <div className="validate-block">
          <div className="validation-coins">
            <label>Validation Coins:</label>
            <input
              type="number"
              min="1"
              max="100"
              value={validationCoins}
              onChange={(e) => setValidationCoins(e.target.value)}
              disabled={isLoading}
            />
          </div>
          
          <div className="block-action-buttons">
            <button
              className="validate-button"
              onClick={validateBlock}
              disabled={isLoading || !blockId}
            >
              {isLoading ? 'Validating...' : 'Validate Block'}
            </button>
            
            <button
              className="analyze-button"
              onClick={analyzeBlock}
              disabled={isLoading || !blockId}
            >
              {isLoading ? 'Analyzing...' : 'Analyze Block'}
            </button>
          </div>
        </div>
        
        {validationResult && (
          <div className="result-box validation-result">
            <h4>Validation Result:</h4>
            <pre>{validationResult}</pre>
          </div>
        )}
        
        {analysisResult && (
          <div className="result-box analysis-result">
            <h4>Analysis Result:</h4>
            <pre>{analysisResult}</pre>
          </div>
        )}
      </div>
      
      {/* Error display */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      <style jsx>{`
        .swap-container {
          max-width: 500px;
          margin: 0 auto;
          padding: 20px;
          font-family: 'Inter', sans-serif;
        }
        
        h2, h3, h4 {
          color: var(--primary-color, #FF0000);
          text-align: center;
          margin-bottom: 20px;
        }
        
        .swap-box, .block-operations {
          background-color: #fff;
          border-radius: 12px;
          padding: 20px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          margin-bottom: 20px;
        }
        
        .swap-input-container {
          margin-bottom: 15px;
        }
        
        .swap-input-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 14px;
        }
        
        .swap-input-group {
          display: flex;
          border: 1px solid #ddd;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .swap-input {
          flex: 1;
          padding: 12px;
          border: none;
          font-size: 16px;
        }
        
        .token-select {
          width: 100px;
          padding: 12px;
          border: none;
          background-color: #f5f5f5;
          font-size: 16px;
          cursor: pointer;
        }
        
        .swap-arrow {
          display: flex;
          justify-content: center;
          margin: 

