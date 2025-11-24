import asyncio
import logging

from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import bleach
import httpx
import socketio
import validators

from a2a.client import A2ACardResolver
from a2a.client.client import Client, ClientConfig, ClientEvent
from a2a.client.client_factory import ClientFactory
from a2a.types import (
    AgentCard,
    FilePart,
    FileWithBytes,
    Message,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskStatusUpdateEvent,
    TextPart,
    TransportProtocol,
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


STANDARD_HEADERS = {
    'host',
    'user-agent',
    'accept',
    'content-type',
    'content-length',
    'connection',
    'accept-encoding',
}

# ==============================================================================
# Setup
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI()
# NOTE: In a production environment, cors_allowed_origins should be restricted
# to the specific frontend domain, not a wildcard '*'.
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount('/socket.io', socket_app)

app.mount('/static', StaticFiles(directory='../frontend/public'), name='static')
templates = Jinja2Templates(directory='../frontend/public')

# ==============================================================================
# State Management
# ==============================================================================

# NOTE: This global dictionary stores state. For a simple inspector tool with
# transient connections, this is acceptable. For a scalable production service,
# a more robust state management solution (e.g., Redis) would be required.
clients: dict[str, tuple[httpx.AsyncClient, Client, AgentCard, str]] = {}

# Task storage: Maps session_id -> {task_id: {contextId, status, lastPolled, artifacts}}
tasks: dict[str, dict[str, dict[str, Any]]] = {}

# Active polling tasks: Maps session_id -> {task_id: asyncio.Task}
polling_tasks: dict[str, dict[str, Any]] = {}


# ==============================================================================
# Socket.IO Event Helpers
# ==============================================================================


async def _emit_debug_log(
    sid: str, event_id: str, log_type: str, data: Any
) -> None:
    """Helper to emit a structured debug log event to the client."""
    await sio.emit(
        'debug_log', {'type': log_type, 'data': data, 'id': event_id}, to=sid
    )


async def _process_a2a_response(
    client_event: ClientEvent | Message,
    sid: str,
    request_id: str,
) -> None:
    """Processes a response from the A2A client, validates it, and emits events.

    This function handles the incoming ClientEvent or Message object,
    correlating it with the original request using the session ID and request ID.

    Args:
    client_event: The event or message received.
    sid: The session ID associated with the original request.
    request_id: The unique ID of the original request.
    """
    # The response payload 'event' (Task, Message, etc.) may have its own 'id',
    # which can differ from the JSON-RPC request/response 'id'. We prioritize
    # the payload's ID for client-side correlation if it exists.

    event: TaskStatusUpdateEvent | TaskArtifactUpdateEvent | Task | Message
    if isinstance(client_event, tuple):
        event = client_event[1] if client_event[1] else client_event[0]
    else:
        event = client_event

    response_id = getattr(event, 'id', request_id)

    response_data = event.model_dump(exclude_none=True)
    response_data['id'] = response_id

    validation_errors = validators.validate_message(response_data)
    response_data['validation_errors'] = validation_errors

    # Store task metadata if this is a Task or TaskStatusUpdateEvent
    if isinstance(event, Task):
        if sid not in tasks:
            tasks[sid] = {}

        # Defensive status extraction
        status_state = 'unknown'
        try:
            if event.status and hasattr(event.status, 'state'):
                status_state = event.status.state
            elif event.status:
                logger.warning(f'Task {event.id} has status but no state attribute')
        except Exception as e:
            logger.warning(f'Could not extract status from Task {event.id}: {e}')

        # Defensive artifacts extraction
        artifacts_list = []
        try:
            if event.artifacts:
                artifacts_list = [art.model_dump(exclude_none=True) for art in event.artifacts]
        except Exception as e:
            logger.warning(f'Could not extract artifacts from Task {event.id}: {e}')

        tasks[sid][event.id] = {
            'id': event.id,
            'contextId': event.context_id,
            'status': status_state,
            'artifacts': artifacts_list,
            'lastPolled': None,
        }

        logger.info(f'✓ Stored Task {event.id[:8]}... [status={status_state}, contextId={event.context_id[:8] if event.context_id else None}..., artifacts={len(artifacts_list)}]')
    elif isinstance(event, TaskStatusUpdateEvent):
        if sid in tasks and event.task_id in tasks[sid]:
            tasks[sid][event.task_id]['status'] = event.task_status.state
            if event.artifacts:
                tasks[sid][event.task_id]['artifacts'] = [
                    art.model_dump(exclude_none=True) for art in event.artifacts
                ]
            logger.info(f'✓ Updated Task {event.task_id[:8]}... [status={event.task_status.state}]')

    await _emit_debug_log(sid, response_id, 'response', response_data)
    await sio.emit('agent_response', response_data, to=sid)


def get_card_resolver(
    client: httpx.AsyncClient, agent_card_url: str
) -> A2ACardResolver:
    """Returns an A2ACardResolver for the given agent card URL."""
    parsed_url = urlparse(agent_card_url)
    base_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
    path_with_query = urlunparse(
        ('', '', parsed_url.path, '', parsed_url.query, '')
    )
    card_path = path_with_query.lstrip('/')
    if card_path:
        card_resolver = A2ACardResolver(
            client, base_url, agent_card_path=card_path
        )
    else:
        card_resolver = A2ACardResolver(client, base_url)

    return card_resolver


# ==============================================================================
# FastAPI Routes
# ==============================================================================


@app.get('/', response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the main index.html page."""
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/agent-card')
async def get_agent_card(request: Request) -> JSONResponse:
    """Fetch and validate the agent card from a given URL."""
    # 1. Parse request and get sid. If this fails, we can't do much.
    try:
        request_data = await request.json()
        agent_url = request_data.get('url')
        sid = request_data.get('sid')

        if not agent_url or not sid:
            return JSONResponse(
                content={'error': 'Agent URL and SID are required.'},
                status_code=400,
            )
    except Exception:
        logger.warning('Failed to parse JSON from /agent-card request.')
        return JSONResponse(
            content={'error': 'Invalid request body.'}, status_code=400
        )

    # Extract custom headers from the request
    custom_headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in STANDARD_HEADERS
    }

    # 2. Log the request.
    await _emit_debug_log(
        sid,
        'http-agent-card',
        'request',
        {
            'endpoint': '/agent-card',
            'payload': request_data,
            'custom_headers': custom_headers,
        },
    )

    # 3. Perform the main action and prepare response.
    try:
        async with httpx.AsyncClient(
            timeout=30.0, headers=custom_headers
        ) as client:
            card_resolver = get_card_resolver(client, agent_url)
            card = await card_resolver.get_agent_card()

        card_data = card.model_dump(exclude_none=True)
        validation_errors = validators.validate_agent_card(card_data)
        response_data = {
            'card': card_data,
            'validation_errors': validation_errors,
        }
        response_status = 200

    except httpx.RequestError as e:
        logger.error(
            f'Failed to connect to agent at {agent_url}', exc_info=True
        )
        response_data = {'error': f'Failed to connect to agent: {e}'}
        response_status = 502  # Bad Gateway
    except Exception as e:
        logger.error('An internal server error occurred', exc_info=True)
        response_data = {'error': f'An internal server error occurred: {e}'}
        response_status = 500

    # 4. Log the response and return it.
    await _emit_debug_log(
        sid,
        'http-agent-card',
        'response',
        {'status': response_status, 'payload': response_data},
    )
    return JSONResponse(content=response_data, status_code=response_status)


# ==============================================================================
# Socket.IO Event Handlers
# ==============================================================================


@sio.on('connect')
async def handle_connect(sid: str, environ: dict[str, Any]) -> None:
    """Handle the 'connect' socket.io event."""
    logger.info(f'Client connected: {sid}, environment: {environ}')


@sio.on('disconnect')
async def handle_disconnect(sid: str) -> None:
    """Handle the 'disconnect' socket.io event."""
    logger.info(f'Client disconnected: {sid}')

    # Cancel all polling tasks for this session
    if sid in polling_tasks:
        for task_id, poll_task in polling_tasks[sid].items():
            poll_task.cancel()
            logger.info(f'Cancelled polling for task {task_id}')
        polling_tasks.pop(sid)

    # Clean up task storage
    if sid in tasks:
        tasks.pop(sid)

    # Clean up client
    if sid in clients:
        httpx_client, _, _, _ = clients.pop(sid)
        await httpx_client.aclose()
        logger.info(f'Cleaned up client for {sid}')


@sio.on('initialize_client')
async def handle_initialize_client(sid: str, data: dict[str, Any]) -> None:
    """Handle the 'initialize_client' socket.io event."""
    agent_card_url = data.get('url')

    custom_headers = data.get('customHeaders', {})

    if not agent_card_url:
        await sio.emit(
            'client_initialized',
            {'status': 'error', 'message': 'Agent URL is required.'},
            to=sid,
        )
        return

    httpx_client = None
    try:
        httpx_client = httpx.AsyncClient(timeout=600.0, headers=custom_headers)
        card_resolver = get_card_resolver(httpx_client, agent_card_url)
        card = await card_resolver.get_agent_card()

        a2a_config = ClientConfig(
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
                TransportProtocol.jsonrpc,
                TransportProtocol.grpc,
            ],
            use_client_preference=True,
            httpx_client=httpx_client,
        )
        factory = ClientFactory(a2a_config)
        a2a_client = factory.create(card)
        transport_protocol = (
            card.preferred_transport or TransportProtocol.jsonrpc
        )

        clients[sid] = (httpx_client, a2a_client, card, transport_protocol)

        input_modes = getattr(card, 'default_input_modes', ['text/plain'])
        output_modes = getattr(card, 'default_output_modes', ['text/plain'])

        streaming_capable = card.capabilities.streaming if card.capabilities else None

        await sio.emit(
            'client_initialized',
            {
                'status': 'success',
                'transport': str(transport_protocol),
                'inputModes': input_modes,
                'outputModes': output_modes,
                'streamingCapable': streaming_capable,
            },
            to=sid,
        )
    except Exception as e:
        logger.error(
            f'Failed to initialize client for {sid}: {e}', exc_info=True
        )
        # Clean up httpx_client
        if httpx_client is not None:
            await httpx_client.aclose()
        await sio.emit(
            'client_initialized', {'status': 'error', 'message': str(e)}, to=sid
        )


@sio.on('send_message')
async def handle_send_message(sid: str, json_data: dict[str, Any]) -> None:
    """Handle the 'send_message' socket.io event."""
    message_text = bleach.clean(json_data.get('message', ''))

    message_id = json_data.get('id', str(uuid4()))
    context_id = json_data.get('contextId')
    metadata = json_data.get('metadata', {})

    if sid not in clients:
        await sio.emit(
            'agent_response',
            {'error': 'Client not initialized.', 'id': message_id},
            to=sid,
        )
        return

    _, a2a_client, _, transport = clients[sid]

    attachments = json_data.get('attachments', [])

    parts: list = []
    if message_text:
        parts.append(TextPart(text=str(message_text)))  # type: ignore[arg-type]

    for attachment in attachments:
        parts.append(
            FilePart(  # type: ignore[arg-type]
                file=FileWithBytes(
                    bytes=attachment['data'], mime_type=attachment['mimeType']
                )
            )
        )

    message = Message(
        role=Role.user,
        parts=parts,
        message_id=message_id,
        context_id=context_id,
        metadata=metadata,
    )

    debug_request = {
        'transport': transport,
        'method': 'message/send',
        'message': message.model_dump(exclude_none=True),
    }
    await _emit_debug_log(sid, message_id, 'request', debug_request)

    try:
        response_stream = a2a_client.send_message(message)
        async for stream_result in response_stream:
            # Log what we received
            event = None
            if isinstance(stream_result, tuple):
                event = stream_result[1] if stream_result[1] else stream_result[0]
            else:
                event = stream_result

            logger.info(f'📩 Received response: {type(event).__name__} (sid={sid[:8]}...)')

            await _process_a2a_response(stream_result, sid, message_id)

            # Auto-start polling for non-streaming agents
            _, _, card, _ = clients[sid]
            if not card.capabilities.streaming:
                if isinstance(event, Task):
                    logger.info(f'🔄 Non-streaming agent detected, auto-starting polling for task {event.id[:8]}...')

                    # Initialize polling_tasks for this session if not exists
                    if sid not in polling_tasks:
                        polling_tasks[sid] = {}

                    # Start polling if not already polling
                    if event.id not in polling_tasks[sid]:
                        poll_task = asyncio.create_task(poll_task_status(sid, event.id))
                        polling_tasks[sid][event.id] = poll_task
                        logger.info(f'⏳ Started polling task {event.id[:8]}... (will check every 3s)')

                        # Notify frontend that polling has started
                        await sio.emit(
                            'polling_status',
                            {'status': 'started', 'taskId': event.id, 'auto': True},
                            to=sid,
                        )
                    else:
                        logger.debug(f'Task {event.id[:8]}... already being polled')
                else:
                    logger.debug(f'Received {type(event).__name__} from non-streaming agent')

    except Exception as e:
        logger.error(f'Failed to send message for sid {sid}', exc_info=True)
        await sio.emit(
            'agent_response',
            {'error': f'Failed to send message: {e}', 'id': message_id},
            to=sid,
        )


@sio.on('get_task')
async def handle_get_task(sid: str, data: dict[str, Any]) -> None:
    """Handle the 'get_task' socket.io event to fetch a single task status."""
    task_id = data.get('taskId')

    if not task_id:
        await sio.emit(
            'task_update',
            {'error': 'Task ID is required.'},
            to=sid,
        )
        return

    if sid not in clients:
        await sio.emit(
            'task_update',
            {'error': 'Client not initialized.'},
            to=sid,
        )
        return

    _, a2a_client, _, _ = clients[sid]

    try:
        query_params = TaskQueryParams(id=task_id)
        task = await a2a_client.get_task(query_params)

        task_data = task.model_dump(exclude_none=True)

        # Update stored task information
        if sid in tasks and task_id in tasks[sid]:
            tasks[sid][task_id]['status'] = task.status.state if task.status else 'unknown'
            tasks[sid][task_id]['artifacts'] = [
                art.model_dump(exclude_none=True) for art in task.artifacts
            ] if task.artifacts else []
            tasks[sid][task_id]['lastPolled'] = asyncio.get_event_loop().time()

        await sio.emit('task_update', task_data, to=sid)

    except Exception as e:
        logger.error(f'Failed to get task {task_id} for sid {sid}', exc_info=True)
        await sio.emit(
            'task_update',
            {'error': f'Failed to get task: {e}', 'taskId': task_id},
            to=sid,
        )


@sio.on('list_tasks')
async def handle_list_tasks(sid: str, data: dict[str, Any]) -> None:
    """Handle the 'list_tasks' socket.io event to list all tasks for a session."""
    context_id = data.get('contextId')

    if sid not in tasks:
        await sio.emit('task_list', {'tasks': [], 'contextId': context_id}, to=sid)
        return

    # Filter tasks by contextId if provided
    session_tasks = tasks[sid]
    if context_id:
        filtered_tasks = {
            tid: task_info
            for tid, task_info in session_tasks.items()
            if task_info.get('contextId') == context_id
        }
    else:
        filtered_tasks = session_tasks

    await sio.emit(
        'task_list',
        {'tasks': list(filtered_tasks.values()), 'contextId': context_id},
        to=sid,
    )


async def poll_task_status(sid: str, task_id: str) -> None:
    """Periodically poll a task's status until it reaches a terminal state."""
    terminal_states = {'completed', 'failed', 'canceled', 'rejected'}
    poll_interval = 3  # seconds

    if sid not in clients:
        logger.warning(f'Cannot poll task {task_id}: client {sid} not found')
        return

    _, a2a_client, _, _ = clients[sid]

    try:
        while True:
            # Check if we should stop polling (task removed or client disconnected)
            if sid not in polling_tasks or task_id not in polling_tasks[sid]:
                logger.info(f'Stopping polling for task {task_id} (sid: {sid})')
                break

            if sid not in clients:
                logger.info(f'Client {sid} disconnected, stopping polling')
                break

            try:
                query_params = TaskQueryParams(id=task_id)
                task = await a2a_client.get_task(query_params)

                task_data = task.model_dump(exclude_none=True)
                current_state = task.status.state if task.status else 'unknown'

                # Check if state changed
                prev_state = tasks[sid][task_id]['status'] if sid in tasks and task_id in tasks[sid] else None
                if prev_state and prev_state != current_state:
                    logger.info(f'Task {task_id} state changed: {prev_state} → {current_state}')
                else:
                    logger.debug(f'Polling task {task_id}, current state: {current_state}')

                # Update stored task information
                if sid in tasks and task_id in tasks[sid]:
                    tasks[sid][task_id]['status'] = current_state
                    tasks[sid][task_id]['artifacts'] = [
                        art.model_dump(exclude_none=True) for art in task.artifacts
                    ] if task.artifacts else []
                    tasks[sid][task_id]['lastPolled'] = asyncio.get_event_loop().time()

                # Emit the updated task
                await sio.emit('task_update', task_data, to=sid)

                # Stop polling if terminal state reached
                if current_state in terminal_states:
                    logger.info(f'Task {task_id} reached terminal state: {current_state}')
                    if sid in polling_tasks and task_id in polling_tasks[sid]:
                        del polling_tasks[sid][task_id]
                    break

            except Exception as e:
                logger.error(f'Error polling task {task_id}: {e}', exc_info=True)
                await sio.emit(
                    'task_update',
                    {'error': f'Polling error: {e}', 'taskId': task_id},
                    to=sid,
                )

            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        logger.info(f'Polling cancelled for task {task_id} (sid: {sid})')
    except Exception as e:
        logger.error(f'Unexpected error in poll_task_status for {task_id}: {e}', exc_info=True)


@sio.on('start_polling_task')
async def handle_start_polling_task(sid: str, data: dict[str, Any]) -> None:
    """Handle the 'start_polling_task' socket.io event to start automatic polling."""
    task_id = data.get('taskId')

    if not task_id:
        await sio.emit(
            'polling_status',
            {'error': 'Task ID is required.', 'taskId': task_id},
            to=sid,
        )
        return

    if sid not in clients:
        await sio.emit(
            'polling_status',
            {'error': 'Client not initialized.', 'taskId': task_id},
            to=sid,
        )
        return

    # Initialize polling_tasks for this session if not exists
    if sid not in polling_tasks:
        polling_tasks[sid] = {}

    # Check if already polling
    if task_id in polling_tasks[sid]:
        await sio.emit(
            'polling_status',
            {'status': 'already_polling', 'taskId': task_id},
            to=sid,
        )
        return

    # Create and store the polling task
    poll_task = asyncio.create_task(poll_task_status(sid, task_id))
    polling_tasks[sid][task_id] = poll_task

    await sio.emit(
        'polling_status',
        {'status': 'started', 'taskId': task_id},
        to=sid,
    )


@sio.on('stop_polling_task')
async def handle_stop_polling_task(sid: str, data: dict[str, Any]) -> None:
    """Handle the 'stop_polling_task' socket.io event to stop automatic polling."""
    task_id = data.get('taskId')

    if not task_id:
        await sio.emit(
            'polling_status',
            {'error': 'Task ID is required.', 'taskId': task_id},
            to=sid,
        )
        return

    if sid in polling_tasks and task_id in polling_tasks[sid]:
        # Cancel the polling task
        poll_task = polling_tasks[sid][task_id]
        poll_task.cancel()
        del polling_tasks[sid][task_id]

        await sio.emit(
            'polling_status',
            {'status': 'stopped', 'taskId': task_id},
            to=sid,
        )
    else:
        await sio.emit(
            'polling_status',
            {'status': 'not_polling', 'taskId': task_id},
            to=sid,
        )


@sio.on('debug_tasks')
async def handle_debug_tasks(sid: str) -> None:
    """Debug endpoint to see what tasks are stored for this session."""
    if sid in tasks:
        logger.info(f'📋 Debug: {len(tasks[sid])} tasks stored for session {sid[:8]}...')
        for task_id, task_info in tasks[sid].items():
            logger.info(f'  Task {task_id[:8]}...: status={task_info["status"]}, contextId={task_info.get("contextId", "None")[:8] if task_info.get("contextId") else "None"}...')

        await sio.emit('debug_log', {
            'type': 'debug',
            'data': {
                'stored_tasks': tasks[sid],
                'task_count': len(tasks[sid]),
                'polling_tasks': list(polling_tasks.get(sid, {}).keys()) if sid in polling_tasks else []
            },
            'id': 'debug-tasks'
        }, to=sid)
    else:
        logger.info(f'📋 Debug: No tasks stored for session {sid[:8]}...')
        await sio.emit('debug_log', {
            'type': 'debug',
            'data': {'stored_tasks': {}, 'task_count': 0, 'message': 'No tasks stored for this session'},
            'id': 'debug-tasks'
        }, to=sid)


# ==============================================================================
# Main Execution
# ==============================================================================


if __name__ == '__main__':
    import uvicorn

    # NOTE: The 'reload=True' flag is for development purposes only.
    # In a production environment, use a proper process manager like Gunicorn.
    uvicorn.run('app:app', host='127.0.0.1', port=5001, reload=True)
