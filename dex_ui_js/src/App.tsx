import React, { useState, useEffect } from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import logo from '../logo.png';
import './App.css';
import Header from './components/Header';
import Footer from './components/Footer';
import Swap from './components/Swap';
import Pool from './components/Pool';
import Farm from './components/Farm';
import { fetchDexConfig } from './utils/connection';

interface DexConfig {
  dex: string;
  ui_path: string;
  logo: string;
}

function App() {
  const [dexConfig, setDexConfig] = useState<DexConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadConfig = async () => {
      try {
        const config = await fetchDexConfig();
        setDexConfig(config);
      } catch (error) {
        console.error('Failed to fetch DEX config:', error);
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  if (loading) {
    return (
      <div className="loading-container">
        <img src={logo} alt="Marble DEX Logo" className="loading-logo" />
        <p style={{ color: '#FF0000' }}>Loading Marble DEX...</p>
      </div>
    );
  }

  return (
    <HashRouter>
      <div className="app">
        <Header logo={logo} title="Marble DEX" />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Swap />} />
            <Route path="/swap" element={<Swap />} />
            <Route path="/pool" element={<Pool />} />
            <Route path="/farm" element={<Farm />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </HashRouter>
  );
}

export default App;

