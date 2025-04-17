import asyncio
from libp2p import Peer, Transport, InboundPeer
from libp2p.mplex import Mplex
from libp2p.streams import Stream

async def run():
    # Crie um peer com transporte TCP
    peer = Peer(Transport.TransportTCP())

    # Inicie o peer
    await peer.start()

    # Obtenha o ID do peer
    peer_id = peer.id

    # Imprima o ID do peer
    print(f"Client ID: {peer_id}")

    # Conecte-se ao servidor
    server_id = "[ID do servidor em hexadecimal]"
    await peer.connect(server_id)

    # Envie dados para o servidor
    stream = await peer.open_stream(server_id, Mplex())
    await send_data(stream)

async def send_data(stream):
    data = b"Hello from Windows!"
    await stream.write(data)
    print(f"Sent data to server: {data.decode('utf-8')}")

if __name__ == "__main__":
    asyncio.run(run())
