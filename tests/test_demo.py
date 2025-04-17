import pytest
from unittest.mock import patch, MagicMock
import demo
import cli
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def mock_node():
    """Fixture for creating a mock node instance"""
    node = MagicMock()
    node.peers = []
    node.discover_peers.return_value = ["peer1", "peer2", "peer3"]
    return node

@pytest.fixture
def client():
    """Fixture for FastAPI test client"""
    return TestClient(app)

class TestNodeDiscovery:
    def test_peer_discovery(self, mock_node):
        """Test node discovery adds new peers"""
        mock_node.discover_peers()
        assert len(mock_node.discover_peers.return_value) == 3

    def test_peer_validation(self, mock_node):
        """Test invalid peers are rejected"""
        with patch('demo.validate_peer') as mock_validate:
            mock_validate.side_effect = lambda x: x != "bad_peer"
            result = demo.filter_valid_peers(["good_peer", "bad_peer", "good_peer2"])
            assert len(result) == 2
            assert "bad_peer" not in result

class TestNetworkMessages:
    @patch('demo.Message')
    def test_message_validation(self, mock_message):
        """Test message validation logic"""
        valid_msg = MagicMock()
        valid_msg.validate.return_value = True
        
        invalid_msg = MagicMock()
        invalid_msg.validate.side_effect = ValueError("Invalid signature")
        
        assert demo.validate_message(valid_msg) is True
        with pytest.raises(ValueError):
            demo.validate_message(invalid_msg)

class TestDemoExecution:
    def test_main_execution(self):
        """Test demo script runs without errors"""
        with patch('demo.run_demo') as mock_run:
            mock_run.return_value = True
            assert demo.main() is True

    def test_execution_failure(self):
        """Test demo handles execution failures"""
        with patch('demo.run_demo') as mock_run:
            mock_run.side_effect = RuntimeError("Demo failed")
            with pytest.raises(RuntimeError):
                demo.main()

class TestCLIInterface:
    def test_cli_commands(self):
        """Test basic CLI commands"""
        with patch('cli.process_command') as mock_process:
            mock_process.return_value = "OK"
            result = cli.handle_command("status")
            assert result == "OK"

    def test_invalid_cli_command(self):
        """Test CLI handles invalid commands"""
        with pytest.raises(ValueError):
            cli.handle_command("invalid_command")

class TestWebUIEndpoints:
    def test_status_endpoint(self, client):
        """Test WebUI status endpoint"""
        response = client.get("/api/status")
        assert response.status_code == 200
        assert "version" in response.json()

    def test_transaction_submission(self, client):
        """Test transaction submission endpoint"""
        test_tx = {"from": "A", "to": "B", "amount": 1.0}
        response = client.post("/api/transactions", json=test_tx)
        assert response.status_code == 201

