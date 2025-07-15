// To use this code, run:
// npm install pdfjs-dist mammoth

import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import ResumeUpload from './components/ResumeUpload';
import RoleSelector from './components/RoleSelector';
import ScoreButton from './components/ScoreButton';
import ScoreDisplay from './components/ScoreDisplay';
import SkillRecommendations from './components/SkillRecommendations';
import Footer from './components/Footer';
import { GlobalWorkerOptions, getDocument, version as pdfjsVersion } from 'pdfjs-dist';
import mammoth from 'mammoth';

const jobRoles = [
  'Software Engineer', 'Data Engineer', 'Product Manager', 'AI Researcher',
  'Frontend Engineer', 'Backend Engineer', 'UI/UX Designer', 'Game Developer',
  'Site Reliability Engineer', 'Platform Engineer', 'Applied Scientist', 'ML Internship',
  'Salesforce SDE', 'Flipkart SDE', 'Amazon SDE', 'Google SDE', 'Meta SDE', 'Microsoft SDE',
  'Apple SDE', 'Netflix SDE', 'Uber SDE', 'Twitter SDE', 'Zomato SDE', 'LinkedIn SDE', 'Tesla SDE',
  'Adobe SDE'
];

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [selectedRole, setSelectedRole] = useState('');
  const [score, setScore] = useState(null);
  const [skills, setSkills] = useState({ recommended: [], matched: [], missing: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Use the worker from the CDN (works for most people)
  GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';

  // Helper: Extract text from file
  const extractText = async (file) => {
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'pdf') {
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await getDocument({ data: arrayBuffer }).promise;
      let text = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map(item => item.str).join(' ') + '\n';
      }
      return text;
    } else if (ext === 'docx') {
      const arrayBuffer = await file.arrayBuffer();
      const { value } = await mammoth.extractRawText({ arrayBuffer });
      return value;
    } else if (ext === 'txt') {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
      });
    } else {
      throw new Error('Unsupported file type. Please upload PDF, DOCX, or TXT.');
    }
  };

  // Backend integration
  const handleScore = async () => {
    setLoading(true);
    setError('');
    setScore(null);
    setSkills({ recommended: [], matched: [], missing: [] });
    try {
      const resumeText = await extractText(resumeFile);
      // You can use a real user ID or email if available
      const student_id = 'frontend-demo-' + Math.random().toString(36).slice(2, 10);
      const response = await fetch('/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_id,
          goal: selectedRole,
          resume_text: resumeText
        })
      });
      if (!response.ok) {
        throw new Error('Failed to score resume.');
      }
      const data = await response.json();
      setScore(Math.round((data.score || 0) * 100));
      setSkills({
        recommended: (data.suggested_learning_path || []).map(item => item.course),
        matched: data.matched_skills || [],
        missing: data.missing_skills || []
      });
    } catch (err) {
      setError(err.message || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <Header />
      <main className="main-card">
        <ResumeUpload onFileSelect={setResumeFile} fileName={resumeFile ? resumeFile.name : ''} />
        <RoleSelector roles={jobRoles} selectedRole={selectedRole} onSelectRole={setSelectedRole} />
        <ScoreButton disabled={!resumeFile || !selectedRole || loading} onClick={handleScore} />
        {loading && <div className="loading">Scoring...</div>}
        {error && <div className="loading" style={{ color: 'red' }}>{error}</div>}
        {score !== null && !loading && !error && (
          <>
            <ScoreDisplay score={score} />
            <SkillRecommendations {...skills} />
          </>
        )}
      </main>
      <Footer />
    </div>
  );
}

export default App;
