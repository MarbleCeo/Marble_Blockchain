import unittest
import asyncio
import os
import sys
import json
from unittest.mock import Mock, patch, AsyncMock

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from networking.client_merged import Client, MessageType
from blockchain.blockchain_merged import Transaction

class TestNetworking(unittest.TestCase):
    """Test cases for the networking components."""
    
    def setUp(self):
        """Set up tests with mocked network components."""
        self.message_handler = Mock()
        self.connection_handler = Mock()
        
        # Create client with mock handlers
        self.client = Client(
            nickname="test_user",
            message_handler=self.message_handler,
            connection_handler=self.connection_handler,
            autostart=False
        )
        
    @patch('networking.client_merged.Client._setup_libp2p_node')
    @patch('networking.client_merged.Client._start_network_tasks')
    async def test_client_startup(self, mock_start_tasks, mock_setup_node):
        """Test client startup sequence."""
        # Setup mock for libp2p node
        mock_setup_node.return_value = AsyncMock()
        
        # Start the client
        await self.client.start()
        
        # Verify the setup was called
        mock_setup_node.assert_called_once()
        mock_start_tasks.assert_called_once()
        
    @patch('networking.client_merged.Client._publish_message')
    async def test_send_chat_message(self, mock_publish):
        """Test sending chat messages."""
        # Setup mock
        mock_publish.return_value = asyncio.Future()
        mock_publish.return_value.set_result(True)
        
        # Send a message
        message = "Hello, world!"
        await self.client.send_chat_message(message)
        
        # Verify the message was published with correct type
        mock_publish.assert_called_once()
        args = mock_publish.call_args[0]
        
        # First arg should be a dictionary with type and content
        self.assertEqual(args[0]["type"], MessageType.CHAT.value)
        self.assertEqual(args[0]["content"], message)
        
    @patch('networking.client_merged.Client._publish_message')
    async def test_send_transaction(self, mock_publish):
        """Test sending transactions."""
        # Setup mock
        mock_publish.return_value = asyncio.Future()
        mock_publish.return_value.set_result(True)
        
        # Create and send a transaction
        transaction = Transaction("sender", "recipient", 10.0)
        await self.client.send_transaction(transaction)
        
        # Verify the transaction was published
        mock_publish.assert_called_once()
        args = mock_publish.call_args[0]
        
        # First arg should be a dictionary with type and transaction data
        self.assertEqual(args[0]["type"], MessageType.TRANSACTION.value)
        self.assertIn("transaction", args[0])
        
    @patch('networking.client_merged.Client._handle_incoming_message')
    async def test_message_handling(self, mock_handler):
        """Test message handling."""
        # Setup test message
        test_message = {
            "type": MessageType.CHAT.value,
            "sender": "other_user",
            "content": "Test message",
            "timestamp": "2023-06-01T12:00:00"
        }
        
        # Create mock for incoming message and call handler
        mock_message = AsyncMock()
        mock_message.data = json.dumps(test_message).encode('utf-8')
        
        # Setup handler return value
        mock_handler.return_value = asyncio.Future()
        mock_handler.return_value.set_result(None)
        
        # Call the method under test
        await self.client._on_pubsub_message(mock_message)
        
        # Verify handler was called with the message
        mock_handler.assert_called_once()
        
    def test_connection_status(self):
        """Test connection status tracking."""
        # Test initial state
        self.assertFalse(self.client.is_connected)
        
        # Test connection status change
        self.client._update_connection_status(True)
        self.assertTrue(self.client.is_connected)
        
        # Verify connection handler was called
        self.connection_handler.assert_called_once_with(True)

if __name__ == '__main__':
    unittest.main()

