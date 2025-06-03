import { useEffect, useRef } from 'react';
import './LogStreamPanel.css';

function cleanLogLine(line) {
  if (line.trim().endsWith('===')) {
    const pipe_index = line.lastIndexOf('|');
    return line.slice(pipe_index + 1).trim();
  }
  const idx = line.lastIndexOf(']');
  if (idx !== -1 && idx < line.length - 1) {
    return line.slice(idx + 1).trim();
  }
  const match = line.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| [A-Z]+ \| ?(\[.*?\] )?(.*)$/);
  if (match) {
    return match[2].trim();
  }
  return line;
}

const getClassName = (line) => {
  if (line.includes('[FeatureAgent]')) return 'log-line feature-agent';
  if (line.includes('[ModelAgent]')) return 'log-line model-agent';
  if (line.includes('[EvaluationAgent]')) return 'log-line evaluation-agent';
  if (line.includes('[Training]')) return 'log-line training';
  if (line.toLowerCase().includes('error')) return 'log-line error';
  return 'log-line default';
};

const LogStreamPanel = ({ logs }) => {
  const logsEndRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="log-panel">
      {logs.map((line, index) => (
        <div key={index} className={getClassName(line)}>
          {cleanLogLine(line)}
        </div>
      ))}
      <div ref={logsEndRef} />
    </div>
  );
};

export default LogStreamPanel;
