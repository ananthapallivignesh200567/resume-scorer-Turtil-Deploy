import React from 'react';
import './ScoreButton.css';

function ScoreButton({ disabled, onClick }) {
  return (
    <button className="score-btn" disabled={disabled} onClick={onClick}>
      Score My Resume
    </button>
  );
}

export default ScoreButton; 