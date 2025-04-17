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
    print(f"Server ID: {peer_id}")

    # Espere por conexões de entrada
    async with peer.listen(port=4001):
        async for inbound_peer in peer.inbound_peers:
            stream = await inbound_peer.open_stream(Mplex())
            await receive_data(stream)

async def receive_data(stream):
    data = await stream.read()
    print(f"Received from {stream.remote_peer.id}: {data.decode('utf-8')}")

if __name__ == "__main__":
    asyncio.run(run())
