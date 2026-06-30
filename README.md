<div align="center">

# Innate Cloud Agent

*Cloud-based inference service for Innate robots*

[![Discord](https://img.shields.io/badge/Discord-Join%20our%20community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/innate)
[![Documentation](https://img.shields.io/badge/Docs-Read%20the%20docs-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://docs.innate.bot)
[![Website](https://img.shields.io/badge/Website-Visit%20us-orange?style=for-the-badge&logo=safari&logoColor=white)](https://innate.bot)

</div>

---

> [!NOTE]
> **This service is in active development.** APIs and features may change. Join our Discord for updates and support.

---

## Overview

The Cloud Agent provides the "brain" for Innate robots — receiving sensor data via WebSocket, processing it through vision-language models, and returning navigation/action commands. It can run locally in Docker or deployed to Google Cloud Run.

## Quick Start

### Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build and run the Docker image

Standard mode (memory commands disabled):
```bash
docker compose -f docker-compose.local.yml build
docker compose -f docker-compose.local.yml up
```

Benchmark mode (memory commands enabled):
```bash
docker compose -f docker-compose.benchmark.yml build
docker compose -f docker-compose.benchmark.yml up
```

### Memory State Management

Cloud Agent includes a memory state management feature that allows saving and loading brain states. This feature is:
- **Disabled** by default in `docker-compose.local.yml`
- **Enabled** by default in `docker-compose.benchmark.yml`

To enable memory state management when running locally:

```bash
python run_server.py --enable-memory-commands
```

#### Memory Commands (when enabled)

When memory state management is enabled, the following commands are available via chat:

- `!save_memory NAME` - Saves the current brain state (history and pose graph memory)
- `!load_memory NAME` - Loads a previously saved brain state
- `!list_memory` - Lists all available saved memory states

You can also provide a `memory_state` parameter in reset messages to load a specific state:

```python
reset_msg = MessageIn(
    type=MessageInType.RESET, payload={"memory_state": "your_state_name"}
)
```

## Cloud Run deployment

Building, pushing, and deploying the image to Google Cloud Run is documented
separately in [docs/cloud.md](docs/cloud.md).

## WebSocket Protocol

The Cloud Agent uses a WebSocket-based protocol for communication between the client (robot) and the server (cloud agent). The protocol consists of a handshake phase followed by an ongoing image exchange and command flow.

### Connection and Handshake Protocol

The handshake protocol establishes the connection and authenticates the client:

```mermaid
sequenceDiagram
    participant Client as Robot Client
    participant Server as Cloud Agent Server
    
    Client->>Server: WebSocket Connection Request
    Server->>Client: Connection Established
    
    Client->>Server: Authentication Message (type: "auth", payload: {"token": "TOKEN"})
    Note over Server: Validate token
    
    alt Authentication Successful
        Server->>Client: Ready for Image (type: "ready_for_image", payload: {})
    else Authentication Failed
        Server->>Client: Close Connection
    end
```

### Image Exchange and Command Flow

After successful authentication, the protocol follows this pattern:

```mermaid
sequenceDiagram
    participant Client as Robot Client
    participant Server as Cloud Agent Server
    participant Brain as Agent Brain
    
    Server->>Client: Ready for Image (type: "ready_for_image", payload: {})
    
    loop Image Processing Cycle
        Client->>Server: Send Image (type: "image", payload: {"image_b64": "...", "depth_map": "...", "robot_coords": {...}})
        Server->>Brain: Process Image
        Note over Brain: Run Visual Language Model
        Brain->>Server: Vision Agent Output
        Server->>Client: Vision Output (type: "vision_agent_output", payload: {"observation": "...", "next_task": {...}})
        Server->>Client: Ready for Image (type: "ready_for_image", payload: {})
        
        alt Primitive Execution
            Client->>Server: Primitive Activated (type: "primitive_activated", payload: {"primitive_name": "..."})
            Note over Client: Execute primitive
            Client->>Server: Primitive Completed/Failed (type: "primitive_completed"/"primitive_failed", payload: {"primitive_name": "..."})
            Server->>Client: Ready for Image (type: "ready_for_image", payload: {})
        end
        
        alt User Chat
            Client->>Server: Chat Message (type: "chat_in", payload: {"text": "..."})
            Note over Brain: Process chat
            opt Fast agent can answer without a new image
                Server->>Client: Chat Output (type: "chat_out", payload: {"text": "..."})
            end
            Server->>Client: Ready for Image (type: "ready_for_image", payload: {})
        end
    end
```

For `chat_in` messages that do not include `image_b64`, the client should wait
for the server's immediate response and the following `ready_for_image` before
sending the next standalone image. If the `chat_in` payload includes `image_b64`,
the cloud agent can run the slow visual agent on that image and return
`vision_agent_output`; clients should still wait for the next readiness signal
before sending additional image frames.

### Message Types

#### Incoming Messages (Client to Server)
- `auth`: Authentication with token
- `image`: Image data with optional depth map and robot coordinates
- `pose_image`: Image data accompanied by pose information
- `reset`: Reset the brain state, optionally loading a saved `memory_state`
- `chat_in`: User chat message
- `primitive_activated`: Notification that a primitive has started execution
- `primitive_completed`: Notification that a primitive has completed successfully
- `primitive_failed`: Notification that a primitive has failed
- `primitive_interrupted`: Notification that a primitive was interrupted
- `primitive_feedback`: Feedback from a primitive during execution
- `register_primitives_and_directive`: Register new primitives and/or directive

#### Outgoing Messages (Server to Client)
- `ready_for_image`: Server is ready to receive a new image
- `vision_agent_output`: Result of processing an image, including observations and next task
- `chat_out`: Chat message from the agent to the user
- `thoughts`: Internal thoughts/reasoning from the agent
- `primitives_and_directive_registered`: Confirmation of primitive/directive registration
- `memory_positions`: Saved memory positions from the pose graph
- `error`: Error message from the server

### Primitive Execution Flow

When the agent decides to execute a primitive:

1. The server sends a `vision_agent_output` with a `next_task` field containing the primitive details
2. The client executes the primitive and sends a `primitive_activated` message
3. After execution, the client sends either `primitive_completed` or `primitive_failed`
4. The server responds with `ready_for_image` to continue the cycle

This protocol enables continuous visual feedback and command execution between the robot client and the cloud agent.
