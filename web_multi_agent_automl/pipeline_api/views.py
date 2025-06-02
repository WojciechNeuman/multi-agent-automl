from django.shortcuts import render
import os
import shutil
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings # To access BASE_DIR if needed for uploads
from django.core.files.storage import FileSystemStorage

# Now this import should work thanks to the sys.path modification in settings.py
from pipeline_controller.pipeline_controller import PipelineController

# Define a directory for temporary uploads within your Django project
# This will be 'multi_agent_automl/web_multi_agent_automl/temp_uploads'
UPLOAD_TEMP_DIR = os.path.join(settings.BASE_DIR, 'temp_uploads')
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
fs = FileSystemStorage(location=UPLOAD_TEMP_DIR)

@csrf_exempt
def run_pipeline_view(request):
    if request.method == 'POST':
        temp_file_path = None # Initialize to ensure it's defined for finally block
        try:
            csv_file = request.FILES.get('csv_file')
            target_column = request.POST.get('target_column')
            problem_type = request.POST.get('problem_type')
            max_iterations_str = request.POST.get('max_iterations')
            main_metric = request.POST.get('main_metric')

            if not csv_file:
                return JsonResponse({'error': 'No CSV file provided.'}, status=400)
            if not all([target_column, problem_type, max_iterations_str, main_metric]):
                return JsonResponse({'error': 'Missing form data.'}, status=400)

            try:
                max_iterations = int(max_iterations_str)
            except ValueError:
                return JsonResponse({'error': 'max_iterations must be an integer.'}, status=400)

            # Save the uploaded file temporarily using FileSystemStorage
            filename = fs.save(csv_file.name, csv_file)
            temp_file_path = fs.path(filename) # Get the full path to the saved file

            controller = PipelineController(
                dataset_path=temp_file_path,
                target_column=target_column,
                problem_type=problem_type,
                max_iterations=max_iterations,
                main_metric=main_metric
            )
            best_result_data = controller.run_full_pipeline()
            pipeline_structure_string = str(controller.pipeline)

            if hasattr(best_result_data.get('model_name'), 'value'):
                best_result_data['model_name'] = best_result_data['model_name'].value

            return JsonResponse({
                'best_result': best_result_data,
                'pipeline_structure': pipeline_structure_string
            })

        except Exception as e:
            print(f"Error during pipeline execution: {e}") # Log to console
            return JsonResponse({'error': f'Pipeline execution failed: {str(e)}'}, status=500)
        finally:
            # Clean up: Remove the temporary file
            if temp_file_path and fs.exists(filename): # filename will be defined if save was successful
                fs.delete(filename)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)