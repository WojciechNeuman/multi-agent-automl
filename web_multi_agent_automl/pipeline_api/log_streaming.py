import queue
import time
from django.http import StreamingHttpResponse

# Global dictionaries to store log queues and results per run
# For a production app with multiple users/workers, you'd need a more robust solution
# like Redis or a database, and proper cleanup mechanisms.
# Key: run_id (str), Value: queue.Queue() for logs
log_queues: dict = {}
# Key: run_id (str), Value: dict (pipeline result or error)
pipeline_results_store: dict = {}

# Special message to signal the end of logs for a stream
END_OF_LOGS_SIGNAL = "END_OF_LOGS_STREAM"

def sse_log_sink_for_loguru(run_id):
    """
    Creates a sink function for Loguru to push messages to a specific run's queue.
    """
    def sink(message):
        if run_id in log_queues:
            # Format the message as you like. Example:
            record = message.record
            formatted_log = f"{record['time'].strftime('%Y-%m-%d %H:%M:%S')} | {record['level'].name} | {record['message']}"
            log_queues[run_id].put(formatted_log)
    return sink

def log_stream_sse_view(request, run_id):
    """
    Streams log messages for a given run_id using Server-Sent Events.
    """
    if run_id not in log_queues:
        # This can happen if the client connects too fast or run_id is invalid
        # Or if the queue was already cleaned up.
        # For now, let's just close the connection.
        # A more robust solution might wait a bit or return an error event.
        def empty_stream():
            yield f"event: error\ndata: Log stream for run_id {run_id} not found or already closed.\n\n"
        return StreamingHttpResponse(empty_stream(), content_type="text/event-stream")


    def event_stream_generator():
        print(f"SSE Stream starting for run_id: {run_id}")
        try:
            q = log_queues[run_id]
            while True:
                try:
                    log_message = q.get(timeout=1) # Wait 1 second for a message
                    if log_message == END_OF_LOGS_SIGNAL:
                        yield f"event: end\ndata: {END_OF_LOGS_SIGNAL}\n\n"
                        break # End of stream
                    yield f"data: {log_message}\n\n"
                except queue.Empty:
                    # Send a keep-alive comment to prevent connection timeout
                    yield ": keepalive\n\n"
                # Add a small delay if needed, but q.get(timeout=1) handles waiting
        except GeneratorExit: # Client disconnected
            print(f"SSE Stream client disconnected for run_id: {run_id}")
        except Exception as e:
            print(f"Error in SSE stream for {run_id}: {e}")
            yield f"event: error\ndata: An error occurred in the log stream: {e}\n\n"
        finally:
            # Clean up the queue for this run_id
            if run_id in log_queues:
                # Empty the queue before deleting to ensure producer (if stuck) doesn't block
                while not log_queues[run_id].empty():
                    try: log_queues[run_id].get_nowait()
                    except queue.Empty: break
                del log_queues[run_id]
            print(f"SSE Stream cleaned up for run_id: {run_id}")

    response = StreamingHttpResponse(event_stream_generator(), content_type="text/event-stream")
    response['Cache-Control'] = 'no-cache' # Ensure no caching
    return response
