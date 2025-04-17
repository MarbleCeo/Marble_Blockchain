import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';

interface HeaderProps {
  logo: string;
}

const HeaderContainer = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: #1a1a1a;
  border-bottom: 2px solid #FF0000;
  position: sticky;
  top: 0;
  z-index: 10;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);

  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

const LogoContainer = styled.div`
  display: flex;
  align-items: center;
`;

const LogoImg = styled.img`
  height: 40px;
  margin-right: 10px;
`;

const LogoText = styled.h1`
  color: #FF0000;
  font-size: 1.5rem;
  font-weight: bold;
  margin: 0;
`;

const NavContainer = styled.nav`
  display: flex;
  align-items: center;

  @media (max-width: 768px) {
    display: none;
  }
`;

const NavItem = styled(Link)`
  color: white;
  margin-left: 1.5rem;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s ease;

  &:hover {
    color: #FF0000;
  }

  &.active {
    color: #FF0000;
  }
`;

const MobileMenuButton = styled.button`
  display: none;
  background: none;
  border: none;
  font-size: 1.5rem;
  color: #FF0000;
  cursor: pointer;

  @media (max-width: 768px) {
    display: block;
  }
`;

const MobileMenu = styled.div<{ isOpen: boolean }>`
  display: ${({ isOpen }) => (isOpen ? 'flex' : 'none')};
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  flex-direction: column;
  background-color: #1a1a1a;
  padding: 1rem;
  border-bottom: 2px solid #FF0000;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
`;

const MobileNavItem = styled(Link)`
  color: white;
  text-decoration: none;
  padding: 0.75rem 0;
  font-weight: 500;
  border-bottom: 1px solid #333;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    color: #FF0000;
  }

  &.active {
    color: #FF0000;
  }
`;

const ConnectWalletButton = styled.button`
  background-color: #FF0000;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  margin-left: 1.5rem;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: #cc0000;
  }

  @media (max-width: 768px) {
    margin: 0.75rem 0 0.25rem;
    width: 100%;
    padding: 0.75rem;
  }
`;

const Header: React.FC<HeaderProps> = ({ logo }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  return (
    <HeaderContainer>
      <LogoContainer>
        <LogoImg src={logo} alt="Marble DEX Logo" />
        <LogoText>Marble DEX</LogoText>
      </LogoContainer>

      <NavContainer>
        <NavItem to="/">Home</NavItem>
        <NavItem to="/swap">Swap</NavItem>
        <NavItem to="/pool">Pool</NavItem>
        <NavItem to="/bridge">Bridge</NavItem>
        <NavItem to="/farm">Farm</NavItem>
        <ConnectWalletButton>Connect Wallet</ConnectWalletButton>
      </NavContainer>

      <MobileMenuButton onClick={toggleMobileMenu}>
        {mobileMenuOpen ? '✕' : '☰'}
      </MobileMenuButton>

      <MobileMenu isOpen={mobileMenuOpen}>
        <MobileNavItem to="/" onClick={() => setMobileMenuOpen(false)}>Home</MobileNavItem>
        <MobileNavItem to="/swap" onClick={() => setMobileMenuOpen(false)}>Swap</MobileNavItem>
        <MobileNavItem to="/pool" onClick={() => setMobileMenuOpen(false)}>Pool</MobileNavItem>
        <MobileNavItem to="/bridge" onClick={() => setMobileMenuOpen(false)}>Bridge</MobileNavItem>
        <MobileNavItem to="/farm" onClick={() => setMobileMenuOpen(false)}>Farm</MobileNavItem>
        <ConnectWalletButton onClick={() => setMobileMenuOpen(false)}>Connect Wallet</ConnectWalletButton>
      </MobileMenu>
    </HeaderContainer>
  );
};

export default Header;

