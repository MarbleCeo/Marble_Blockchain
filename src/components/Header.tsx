import React from 'react';
import './Header.css';

interface HeaderProps {
  walletAddress?: string;
  onConnectWallet: () => void;
}

const Header: React.FC<HeaderProps> = ({ walletAddress, onConnectWallet }) => {
  return (
    <header className="header">
      <div className="container header-container">
        <div className="logo">
          <span className="logo-icon">M</span>
          <h1>Marble DEX</h1>
        </div>
        <nav className="navigation">
          <ul>
            <li><a href="#" className="active">Swap</a></li>
            <li><a href="#">Liquidity</a></li>
            <li><a href="#">Farms</a></li>
            <li><a href="#">Staking</a></li>
          </ul>
        </nav>
        <div className="wallet-section">
          {walletAddress ? (
            <div className="wallet-info">
              <span className="wallet-address">
                {`${walletAddress.substring(0, 4)}...${walletAddress.substring(walletAddress.length - 4)}`}
              </span>
            </div>
          ) : (
            <button className="connect-wallet" onClick={onConnectWallet}>
              Connect Wallet
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;

