import React, { useRef } from 'react';
import './ResumeUpload.css';

function ResumeUpload({ onFileSelect, fileName }) {
  const fileInputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="resume-upload" onDrop={handleDrop} onDragOver={e => e.preventDefault()}>
      <div className="upload-area" onClick={() => fileInputRef.current.click()}>
        <span role="img" aria-label="upload" className="upload-icon">📄</span>
        <p>{fileName ? fileName : 'Drag & drop your resume here or click to browse'}</p>
        <input
          type="file"
          accept=".pdf,.doc,.docx,.txt"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>
    </div>
  );
}

export default ResumeUpload; 