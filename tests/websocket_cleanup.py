# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

from contextlib import suppress


_TRACKED_SERVERS = []
_TRACKED_CLIENTS = []


def track_websocket_server(server):
    _TRACKED_SERVERS.append(server)
    return server


def track_websocket_client(client):
    _TRACKED_CLIENTS.append(client)
    return client


async def close_tracked_websockets():
    while _TRACKED_CLIENTS:
        client = _TRACKED_CLIENTS.pop()
        with suppress(Exception):
            await client.close()

    while _TRACKED_SERVERS:
        server = _TRACKED_SERVERS.pop()
        with suppress(Exception):
            server.close()
            await server.wait_closed()
