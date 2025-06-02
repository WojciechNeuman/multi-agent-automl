// frontend_ui/src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // We'll create this file next

function App() {
    // State for form inputs
    const [csvFile, setCsvFile] = useState(null);
    const [targetColumn, setTargetColumn] = useState('Survived'); // Default value
    const [problemType, setProblemType] = useState('classification'); // Default value
    const [maxIterations, setMaxIterations] = useState(3); // Default value
    const [mainMetric, setMainMetric] = useState('accuracy'); // Default value

    // State for API response
    const [bestResult, setBestResult] = useState(null);
    const [pipelineStructure, setPipelineStructure] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [fileName, setFileName] = useState('');

    // Handle file input change
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setCsvFile(file);
            setFileName(file.name); // Store file name for display
        } else {
            setCsvFile(null);
            setFileName('');
        }
    };

    // Handle form submission
    const handleSubmit = async (event) => {
        event.preventDefault(); // Prevent default form submission

        if (!csvFile) {
            setError('Please select a CSV file to upload.');
            return;
        }

        setIsLoading(true);
        setError('');
        setBestResult(null);
        setPipelineStructure('');

        // Create FormData object to send file and other data
        const formData = new FormData();
        formData.append('csv_file', csvFile);
        formData.append('target_column', targetColumn);
        formData.append('problem_type', problemType);
        formData.append('max_iterations', maxIterations);
        formData.append('main_metric', mainMetric);

        try {
            // API endpoint URL (ensure your Django server is running and accessible)
            const apiUrl = 'http://localhost:8000/api/run-pipeline/';
            const response = await axios.post(apiUrl, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data', // Important for file uploads
                },
            });

            // Set results from the API response
            setBestResult(response.data.best_result);
            setPipelineStructure(response.data.pipeline_structure);

        } catch (err) {
            // Handle errors from the API call
            const errorMessage = err.response?.data?.error || err.message || 'An unknown error occurred during the pipeline execution.';
            setError(`Error: ${errorMessage}`);
            console.error("Pipeline execution error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Multiagent AutoML</h1>
                <p>Upload your dataset and configure the pipeline parameters.</p>
            </header>

            <main className="App-main">
                <form onSubmit={handleSubmit} className="pipeline-form">
                    {/* File Upload */}
                    <div className="form-group">
                        <label htmlFor="csvFile">Upload CSV File:</label>
                        <input
                            type="file"
                            id="csvFile"
                            accept=".csv"
                            onChange={handleFileChange}
                            required
                        />
                        {fileName && <span className="file-name-display">Selected file: {fileName}</span>}
                    </div>

                    {/* Target Column Input */}
                    <div className="form-group">
                        <label htmlFor="targetColumn">Target Column Name:</label>
                        <input
                            type="text"
                            id="targetColumn"
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            placeholder="e.g., Survived"
                            required
                        />
                    </div>

                    {/* Problem Type Dropdown */}
                    <div className="form-group">
                        <label htmlFor="problemType">Problem Type:</label>
                        <select id="problemType" value={problemType} onChange={(e) => setProblemType(e.target.value)}>
                            <option value="classification">Classification</option>
                            <option value="regression">Regression</option>
                            {/* Add other problem types if your backend supports them */}
                        </select>
                    </div>

                    {/* Max Iterations Dropdown */}
                    <div className="form-group">
                        <label htmlFor="maxIterations">Max Iterations:</label>
                        <select id="maxIterations" value={maxIterations} onChange={(e) => setMaxIterations(parseInt(e.target.value))}>
                            <option value={1}>1</option>
                            <option value={2}>2</option>
                            <option value={3}>3</option>
                            <option value={5}>5</option>
                            <option value={10}>10</option>
                            <option value={20}>20</option>
                        </select>
                    </div>

                    {/* Main Metric Dropdown */}
                    <div className="form-group">
                        <label htmlFor="mainMetric">Main Metric:</label>
                        <select id="mainMetric" value={mainMetric} onChange={(e) => setMainMetric(e.target.value)}>
                            {/* Common metrics */}
                            <option value="accuracy">Accuracy</option>
                            <option value="precision">Precision</option>
                            <option value="recall">Recall</option>
                            <option value="f1_score">F1-score</option>
                            <option value="roc_auc">ROC AUC</option>
                            {/* Regression specific metrics */}
                            <option value="mse">MSE (Mean Squared Error)</option>
                            <option value="mae">MAE (Mean Absolute Error)</option>
                            <option value="r2_score">R2 Score</option>
                            {/* Add other metrics as supported by your backend */}
                        </select>
                    </div>

                    <button type="submit" disabled={isLoading} className="submit-button">
                        {isLoading ? '🧠 Processing Pipeline...' : '🚀 Run Pipeline'}
                    </button>
                </form>

                {/* Display Error Messages */}
                {error && (
                    <div className="error-message">
                        <p>{error}</p>
                    </div>
                )}

                {/* Display Loading State */}
                {isLoading && (
                    <div className="loading-indicator">
                        <p>Pipeline is running, please wait...</p>
                        {/* You can add a spinner or a more visual loading indicator here */}
                    </div>
                )}

                {/* Display Results */}
                {!isLoading && bestResult && (
                    <div className="results-section">
                        <h2>🏆 Best Result:</h2>
                        <pre className="results-output">{JSON.stringify(bestResult, null, 2)}</pre>
                    </div>
                )}

                {!isLoading && pipelineStructure && (
                    <div className="results-section">
                        <h2>🛠️ Pipeline Structure:</h2>
                        <pre className="results-output">{pipelineStructure}</pre>
                    </div>
                )}
            </main>

            <footer className="App-footer">
                <p>&copy; {new Date().getFullYear()} Multi-Agent AutoML Project</p>
            </footer>
        </div>
    );
}

export default App;
