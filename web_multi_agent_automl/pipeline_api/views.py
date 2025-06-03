import queue
import os
import uuid
import threading # For background tasks
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from pipeline_controller.pipeline_controller import PipelineController # Your import
from .log_streaming import log_queues, pipeline_results_store, sse_log_sink_for_loguru, END_OF_LOGS_SIGNAL

from loguru import logger as loguru_logger # Assuming PipelineController uses loguru

# Directory for temporary file uploads & logs (ensure this dir exists or is created)
UPLOAD_TEMP_DIR = os.path.join(settings.BASE_DIR, 'temp_pipeline_files')
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
fs = FileSystemStorage(location=UPLOAD_TEMP_DIR)


def run_pipeline_background_logic(run_id, dataset_file_path, target_column, problem_type, max_iterations, main_metric):
    """
    This function contains the actual pipeline logic and runs in a background thread.
    It will log to a queue associated with run_id.
    """
    sink_id = None
    try:
        # 1. Setup Loguru sink for this run
        # Ensure a queue exists for this run_id before adding the sink
        if run_id not in log_queues: # Should have been created by the main view
            log_queues[run_id] = queue.Queue()
            
        log_sink_func = sse_log_sink_for_loguru(run_id)
        sink_id = loguru_logger.add(log_sink_func, level="INFO") # Adjust level as needed

        loguru_logger.info(f"[{run_id}] Pipeline process started.")
        loguru_logger.info(f"[{run_id}] Dataset: {os.path.basename(dataset_file_path)}, Target: {target_column}")

        # 2. Initialize and run your PipelineController
        controller = PipelineController(
            dataset_path=dataset_file_path,
            target_column=target_column,
            problem_type=problem_type,
            max_iterations=max_iterations,
            main_metric=main_metric
        )
        best_result_data = controller.run_full_pipeline() # This is the blocking call
        pipeline_structure_string = str(controller.pipeline)

        if hasattr(best_result_data.get('model_name'), 'value'):
            best_result_data['model_name'] = best_result_data['model_name'].value
        
        loguru_logger.info(f"[{run_id}] Pipeline process completed successfully.")
        pipeline_results_store[run_id] = {
            'status': 'completed',
            'best_result': best_result_data,
            'pipeline_structure': pipeline_structure_string
        }

    except Exception as e:
        loguru_logger.error(f"[{run_id}] Pipeline execution failed: {str(e)}")
        pipeline_results_store[run_id] = {'status': 'failed', 'error': str(e)}
    finally:
        # 3. Signal end of logs and clean up sink
        if run_id in log_queues:
            log_queues[run_id].put(END_OF_LOGS_SIGNAL)
        if sink_id is not None:
            loguru_logger.remove(sink_id)
        
        # 4. Clean up the temporary dataset file
        if dataset_file_path and os.path.exists(dataset_file_path):
            try:
                os.remove(dataset_file_path)
                loguru_logger.info(f"[{run_id}] Cleaned up temporary dataset file: {dataset_file_path}")
            except Exception as e_remove:
                loguru_logger.error(f"[{run_id}] Error cleaning up temp file {dataset_file_path}: {e_remove}")


@csrf_exempt
def run_pipeline_start_view(request): # Renamed for clarity
    if request.method == 'POST':
        run_id = str(uuid.uuid4())
        temp_file_path_for_thread = None

        try:
            csv_file = request.FILES.get('csv_file')
            target_column = request.POST.get('target_column')
            problem_type = request.POST.get('problem_type')
            max_iterations_str = request.POST.get('max_iterations')
            main_metric = request.POST.get('main_metric')

            if not csv_file:
                return JsonResponse({'error': 'No CSV file provided.'}, status=400)
            # ... (add other validations as before) ...
            max_iterations = int(max_iterations_str)

            # Save uploaded file; its path will be passed to the thread
            # The thread will be responsible for deleting it.
            filename = fs.save(f"{run_id}_{csv_file.name}", csv_file) # Prefix with run_id for uniqueness
            temp_file_path_for_thread = fs.path(filename)

            # Initialize the log queue for this run_id *before* starting the thread
            log_queues[run_id] = queue.Queue()
            pipeline_results_store[run_id] = {'status': 'pending'} # Initial status

            # Start pipeline logic in a background thread
            thread = threading.Thread(
                target=run_pipeline_background_logic,
                args=(run_id, temp_file_path_for_thread, target_column, problem_type, max_iterations, main_metric)
            )
            thread.daemon = True # Allows main program to exit even if threads are running
            thread.start()

            return JsonResponse({'run_id': run_id, 'status': 'started'})

        except Exception as e:
            # If file was saved before error, clean it up
            if temp_file_path_for_thread and fs.exists(filename):
                 fs.delete(filename)
            return JsonResponse({'error': f'Failed to start pipeline: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)


@csrf_exempt
def get_pipeline_result_view(request, run_id):
    result = pipeline_results_store.get(run_id)
    if result:
        if result['status'] == 'completed' or result['status'] == 'failed':
            # Optionally remove from store after fetching to save memory,
            # but then it can't be fetched again. Consider a timeout.
            # For now, let's keep it.
            return JsonResponse(result)
        else: # pending or running
            return JsonResponse({'status': result.get('status', 'running')})
    else:
        return JsonResponse({'status': 'not_found', 'error': 'No result found for this run_id.'}, status=404)
