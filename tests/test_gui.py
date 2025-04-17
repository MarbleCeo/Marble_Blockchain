import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock PyQt6 modules since tests might run in environments without display
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()

# Import GUI after mocking
from gui.clientgui_merged import P2PChatApp, AsyncHelper

class TestGUI(unittest.TestCase):
    """Test cases for the GUI components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock for QApplication
        self.app_mock = Mock()
        
        # Create test instance with mocked components
        with patch('gui.clientgui_merged.QApplication', return_value=self.app_mock):
            with patch('gui.clientgui_merged.Client'):
                self.gui = P2PChatApp()
                
                # Mock UI components
                self.gui.chat_text = Mock()
                self.gui.message_input = Mock()
                self.gui.send_button = Mock()
                self.gui.connect_button = Mock()
                self.gui.status_label = Mock()
                self.gui.peers_list = Mock()
                self.gui.transaction_amount = Mock()
                self.gui.sender_address = Mock()
                self.gui.recipient_address = Mock()
                self.gui.balance_label = Mock()
                self.gui.nickname_input = Mock()
                
    def test_initialization(self):
        """Test GUI initialization."""
        # Verify client is initialized
        self.assertIsNotNone(self.gui.client)
        
    def test_send_chat_message(self):
        """Test sending chat messages."""
        # Mock the client's send_chat_message method
        self.gui.client.send_chat_message = AsyncMock = Mock()
        
        # Mock user input
        self.gui.message_input.text.return_value = "Test message"
        
        # Call the method
        self.gui.send_chat_message()
        
        # Verify client method was called
        self.assertEqual(self.gui.client.send_chat_message.call_count, 1)
        
        # Verify message input was cleared
        self.gui.message_input.clear.assert_called_once()
        
    def test_connect_to_network(self):
        """Test network connection functionality."""
        # Mock client's start method
        self.gui.client.start = Mock()
        self.gui.nickname_input.text.return_value = "TestUser"
        
        # Call the method
        self.gui.connect_to_network()
        
        # Verify client was started
        self.gui.client.start.assert_called_once()
        
        # Verify UI was updated
        self.gui.connect_button.setEnabled.assert_called_with(False)
        
    def test_handle_connection_change(self):
        """Test connection status handling."""
        # Test connected state
        self.gui.handle_connection_change(True)
        self.gui.status_label.setText.assert_called_with("Connected")
        
        # Test disconnected state
        self.gui.handle_connection_change(False)
        self.gui.status_label.setText.assert_called_with("Disconnected")
        
    def test_send_transaction(self):
        """Test transaction sending."""
        # Mock form inputs
        self.gui.sender_address.text.return_value = "sender_address"
        self.gui.recipient_address.text.return_value = "recipient_address"
        self.gui.transaction_amount.value.return_value = 10.0
        
        # Mock client's send_transaction method
        self.gui.client.send_transaction = Mock()
        
        # Call the method
        with patch('gui.clientgui_merged.Transaction') as mock_transaction:
            self.gui.send_transaction()
            
            # Verify transaction was created and sent
            mock_transaction.assert_called_once_with(
                "sender_address", "recipient_address", 10.0
            )
            self.assertEqual(self.gui.client.send_transaction.call_count, 1)
            
    def test_handle_message(self):
        """Test message handling."""
        # Create test message
        test_message = {
            "type": "CHAT",
            "sender": "other_user",
            "content": "Test message",
            "timestamp": "2023-06-01T12:00:00"
        }
        
        # Call the method
        self.gui.handle_message(test_message)
        
        # Verify chat was updated
        self.gui.chat_text.append.assert_called_once()
        
    def test_async_helper(self):
        """Test AsyncHelper functionality."""
        # Create test coroutine
        async def test_coro():
            return "test_result"
            
        # Create callback mock
        callback = Mock()
        
        # Create AsyncHelper
        helper = AsyncHelper(test_coro(), callback)
        
        # Verify callback is registered
        self.assertEqual(helper.callback, callback)
        
if __name__ == '__main__':
    unittest.main()

