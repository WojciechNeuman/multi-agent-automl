import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import LogStreamPanel from './components/LogStreamPanel';

const MetricsDisplay = ({ metrics }) => {
    if (!metrics || typeof metrics !== 'object') {
        return <p>No metrics data available.</p>;
    }
    return (
        <ul className="metrics-list">
            {Object.entries(metrics).map(([key, value]) => (
                <li key={key}>
                    <span className="metric-key">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                    <span className="metric-value">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                </li>
            ))}
        </ul>
    );
};

const ListDisplay = ({ items, title }) => {
    if (!items || !Array.isArray(items) || items.length === 0) {
        return <p>No {title.toLowerCase()} data available.</p>;
    }
    return (
        <div className="list-display-container">
            <h4>{title}:</h4>
            <ul className="feature-list">
                {items.map((item, index) => (
                    <li key={index}>{item}</li>
                ))}
            </ul>
        </div>
    );
};


function App() {
    const [csvFile, setCsvFile] = useState(null);
    const [targetColumn, setTargetColumn] = useState('Survived');
    const [problemType, setProblemType] = useState('classification');
    const [maxIterations, setMaxIterations] = useState(3);
    const [mainMetric, setMainMetric] = useState('accuracy');
    const [fileName, setFileName] = useState('');

    const [runId, setRunId] = useState(null);
    const [isPipelineRunning, setIsPipelineRunning] = useState(false);
    const [logs, setLogs] = useState([]);
    const [error, setError] = useState('');
    
    const [bestResult, setBestResult] = useState(null);
    const [pipelineStructure, setPipelineStructure] = useState('');
    const [pipelineHtml, setPipelineHtml] = useState('');

    const eventSourceRef = useRef(null);
    const logsEndRef = useRef(null);

    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [logs]);

    const clearPreviousRunState = () => {
        setLogs([]);
        setError('');
        setBestResult(null);
        setPipelineStructure('');
        setRunId(null);
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setCsvFile(file);
            setFileName(file.name);
        } else {
            setCsvFile(null);
            setFileName('');
        }
    };

    const fetchPipelineResults = async (currentRunId) => {
        if (!currentRunId) return;
        try {
            setLogs(prev => [...prev, "Fetching final results..."]);
            const resultApiUrl = `http://localhost:8000/api/pipeline-result/${currentRunId}/`;
            const response = await axios.get(resultApiUrl);

            if (response.data.status === 'completed') {
                setBestResult(response.data.best_result);
                setPipelineStructure(response.data.pipeline_structure);
                setError('');
            } else if (response.data.status === 'failed') {
                setError(`Pipeline failed: ${response.data.error || 'Unknown error'}`);
                setLogs(prev => [...prev, `Pipeline failed: ${response.data.error || 'Unknown error'}`]);
            } else {
                 setError(`Unknown status from result endpoint: ${response.data.status}`);
            }
        } catch (err) {
            const errorMessage = err.response?.data?.error || err.message || 'Could not fetch pipeline results.';
            setError(`Error fetching results: ${errorMessage}`);
            setLogs(prev => [...prev, `Error fetching results: ${errorMessage}`]);
        } finally {
            setIsPipelineRunning(false);
        }
    };

    useEffect(() => {
        if (runId && isPipelineRunning) {
            const logStreamUrl = `http://localhost:8000/api/log-stream/${runId}/`;
            eventSourceRef.current = new EventSource(logStreamUrl);

            eventSourceRef.current.onmessage = (event) => {
                setLogs(prevLogs => [...prevLogs, event.data]);
            };

            eventSourceRef.current.addEventListener('end', (event) => {
                if (eventSourceRef.current) {
                    eventSourceRef.current.close();
                    eventSourceRef.current = null;
                }
                fetchPipelineResults(runId);
            });
            
            eventSourceRef.current.addEventListener('error', (event) => {
                if (eventSourceRef.current && eventSourceRef.current.readyState === EventSource.CLOSED) {
                    setLogs(prevLogs => [...prevLogs, "Log stream connection closed."]);
                } else if (event.target.readyState === EventSource.CONNECTING) {
                     setLogs(prevLogs => [...prevLogs, "Reconnecting to log stream... (If this persists, check server)"]);
                } else {
                    setLogs(prevLogs => [...prevLogs, "Error with log stream. Closing connection."]);
                    console.error('EventSource failed:', event);
                    if (eventSourceRef.current) {
                        eventSourceRef.current.close();
                        eventSourceRef.current = null;
                    }
                    setError("Log stream error. Attempting to fetch results if possible.");
                    fetchPipelineResults(runId); 
                }
            });
        } else {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        }
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        };
    }, [runId, isPipelineRunning]);

useEffect(() => {
    const fetchHtml = async () => {
        if (runId && !isPipelineRunning) {
            try {
                const response = await axios.get(`http://localhost:8000/api/pipeline-diagram/${runId}/`);
                if (response.data.status === 'ok') {
                    setPipelineHtml(response.data.html);
                }
            } catch (err) {
                console.error('Failed to fetch pipeline diagram:', err);
            }
        }
    };
    fetchHtml();
}, [runId, isPipelineRunning]);

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!csvFile) {
            setError('Please select a CSV file.');
            return;
        }
        clearPreviousRunState();
        setIsPipelineRunning(true);

        const formDataObj = new FormData();
        formDataObj.append('csv_file', csvFile);
        formDataObj.append('target_column', targetColumn);
        formDataObj.append('problem_type', problemType);
        formDataObj.append('max_iterations', maxIterations);
        formDataObj.append('main_metric', mainMetric);

        try {
            const startApiUrl = 'http://localhost:8000/api/start-pipeline/';
            const response = await axios.post(startApiUrl, formDataObj, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            if (response.data.run_id && response.data.status === 'started') {
                setRunId(response.data.run_id);
            } else {
                setError(response.data.error || 'Failed to start pipeline.');
                setIsPipelineRunning(false);
            }
        } catch (err) {
            const errorMessage = err.response?.data?.error || err.message || 'Could not start pipeline execution.';
            setError(`Error starting pipeline: ${errorMessage}`);
            setIsPipelineRunning(false);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>AutoML Pipeline Interface</h1>
                <p>Upload your dataset and configure the pipeline parameters for real-time logging.</p>
            </header>

            <main className="App-main">
                <form onSubmit={handleSubmit} className="pipeline-form">
                     <div className="form-group">
                        <label htmlFor="csvFile">Upload CSV File:</label>
                        <input type="file" id="csvFile" accept=".csv" onChange={handleFileChange} disabled={isPipelineRunning} required />
                        {fileName && <span className="file-name-display">Selected file: {fileName}</span>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="targetColumn">Target Column Name:</label>
                        <input type="text" id="targetColumn" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} disabled={isPipelineRunning} placeholder="e.g., Survived" required />
                    </div>
                    <div className="form-group">
                        <label htmlFor="problemType">Problem Type:</label>
                        <select id="problemType" value={problemType} onChange={(e) => setProblemType(e.target.value)} disabled={isPipelineRunning}>
                            <option value="classification">Classification</option>
                            <option value="regression">Regression</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="maxIterations">Max Iterations:</label>
                        <select id="maxIterations" value={maxIterations} onChange={(e) => setMaxIterations(parseInt(e.target.value))} disabled={isPipelineRunning}>
                            <option value={1}>1</option><option value={2}>2</option><option value={3}>3</option><option value={5}>5</option><option value={10}>10</option><option value={20}>20</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="mainMetric">Main Metric:</label>
                        <select id="mainMetric" value={mainMetric} onChange={(e) => setMainMetric(e.target.value)} disabled={isPipelineRunning}>
                            <option value="accuracy">Accuracy</option><option value="precision">Precision</option><option value="recall">Recall</option><option value="f1_score">F1-score</option><option value="roc_auc">ROC AUC</option>
                            <option value="rmse">RMSE (for regression)</option><option value="mae">MAE (for regression)</option><option value="r2_score">R2 Score</option>
                        </select>
                    </div>
                    <button type="submit" disabled={isPipelineRunning} className="submit-button">
                        {isPipelineRunning ? '🧠 Processing Pipeline...' : '🚀 Run Pipeline'}
                    </button>
                </form>

                {error && <div className="error-message"><p>{error}</p></div>}

                {(logs.length > 0 || isPipelineRunning) && (
                    <div className="logs-section results-section">
                        <h2>Pipeline Logs</h2>
                        <LogStreamPanel logs={logs} />
                    </div>
                )}

                {!isPipelineRunning && bestResult && (
                    <div className="results-section best-result-details"> {}
                        <h2>Best Result</h2>
                        <div className="result-content-padding"> {}
                            <p><strong>Model:</strong> {bestResult.model_name || 'N/A'}</p>
                            
                            {bestResult.hyperparameters && Object.keys(bestResult.hyperparameters).length > 0 && (
                                <div className="hyperparameters-display">
                                    <h4>Hyperparameters:</h4>
                                    <pre className="nested-json-display">
                                        {JSON.stringify(bestResult.hyperparameters, null, 2)}
                                    </pre>
                                </div>
                            )}
                             {bestResult.hyperparameters && Object.keys(bestResult.hyperparameters).length === 0 && (
                                <p><strong>Hyperparameters:</strong> Default</p>
                            )}


                            {bestResult.metrics && <MetricsDisplay metrics={bestResult.metrics} />}
                            
                            {bestResult.features && <ListDisplay items={bestResult.features} title="Selected Features" />}

                            {bestResult.reasoning && (
                                <div className="reasoning-display">
                                    <h4>Reasoning:</h4>
                                    <p>{bestResult.reasoning}</p>
                                </div>
                            )}
                            {bestResult.recommendation && <p><strong>Recommendation:</strong> {bestResult.recommendation}</p>}
                        </div>
                    </div>
                )}

                {!isPipelineRunning && pipelineStructure && (
                    <div className="results-section pipeline-structure-details"> {}
                        <h2>Pipeline Structure</h2>
                        {}
                        <pre className="results-output pipeline-structure-output">
                            {pipelineStructure}
                        </pre>
                    </div>
                )}

                {!isPipelineRunning && pipelineHtml && (
                    <div className="results-section pipeline-visualization">
                        <h2>Pipeline Visualization</h2>
                        <iframe
                            srcDoc={pipelineHtml}
                            style={{ width: '100%', height: '600px', border: '1px solid #ccc' }}
                            title="Pipeline Diagram"
                        />
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
