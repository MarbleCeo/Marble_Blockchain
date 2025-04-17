import socket
import threading
import time

class Client(threading.Thread):
    """
    Represents a connected client on the server.
    """
    def __init__(self, client_socket, client_address, server):
        super().__init__()
        self.socket = client_socket
        self.address = client_address
        self.server = server
        self.nickname = None

    def run(self):
        print(f'Client connected from {self.address}')
        self.receive_messages()

    def receive_messages(self):
        while True:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break

                # Extract nickname (if not set yet)
                if not self.nickname and data.startswith('/nickname '):
                    self.nickname = data.split()[1]
                    self.server.broadcast(f'{self.nickname} has joined the chat.')
                    continue

                # Handle regular messages
                if self.nickname:
                    message = f'{self.nickname}: {data}'
                    self.server.broadcast(message)
                else:
                    print(f'Client {self.address} sent a message without a nickname. Ignoring.')

            except Exception as err:
                print(f'Client error {err}')
                self.server.remove_client(self)
                break

        self.socket.close()
        print(f'Client disconnected: {self.address}')

class Server:
    """
    Represents a chat server that manages client connections.
    """
    def __init__(self, host='127.0.0.1', port=2555):
        self.host = host
        self.port = port
        self.clients = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f'Server is listening on {self.host}:{self.port}')

    def start(self):
        """
        Starts the server and listens for new connections.
        """
        print('Server started.')
        while True:
            client_socket, client_address = self.server_socket.accept()
            client = Client(client_socket, client_address, self)
            client.start()
            self.clients.append(client)

    def stop(self):
        """
        Gracefully stops the server by closing client connections and the server socket.
        """
        print('Server stopping...')
        for client in self.clients:
            client.socket.close()
        self.server_socket.close()
        print('Server stopped.')

    def broadcast(self, message):
        """
        Broadcasts a message to all connected clients.
        """
        for client in self.clients:
            try:
                client.socket.sendall(message.encode('utf-8'))
            except Exception as err:
                print(f'Error sending message to client {client.address}: {err}')
                self.remove_client(client)

    def remove_client(self, client):
        """
        Removes a client from the server's list and closes their socket.
        """
        if client in self.clients:
            self.clients.remove(client)
            client.socket.close()
            self.broadcast(f'{client.nickname} has left the chat.')

if __name__ == '__main__':
    server = Server()
    server.start()