import React from 'react';
import Header from './components/Header';
import Swap from './components/Swap';
import './App.css';

const App: React.FC = () => {
  const [walletAddress, setWalletAddress] = React.useState<string | undefined>(undefined);

  // Mock wallet connection function
  const connectWallet = async () => {
    // In a real application, this would connect to a wallet
    setWalletAddress('DEMO_WALLET_ADDRESS');
  };

  return (
    <div className="app">
      <Header 
        walletAddress={walletAddress} 
        onConnectWallet={connectWallet} 
      />
      <main className="container">
        <h1>Welcome to Marble DEX</h1>
        <p>The next generation decentralized exchange</p>
        <Swap walletAddress={walletAddress} />
      </main>
      <footer>
        <div className="container">
          <p>© 2023 Marble DEX. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;

