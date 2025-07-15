import React from 'react';
import './ScoreDisplay.css';

function ScoreDisplay({ score }) {
  const radius = 60;
  const stroke = 10;
  const normalizedRadius = radius - stroke * 0.5;
  const circumference = normalizedRadius * 2 * Math.PI;
  const progress = score / 100;
  const strokeDashoffset = circumference - progress * circumference;

  return (
    <div className="score-display">
      <svg height={radius * 2} width={radius * 2}>
        <circle
          stroke="#e0e0e0"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <circle
          stroke="#4f8cff"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference + ' ' + circumference}
          style={{ strokeDashoffset, transition: 'stroke-dashoffset 1s ease' }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dy=".3em"
          fontSize="2em"
          fill="#222"
        >
          {score}
        </text>
      </svg>
      <p className="score-label">Your resume matches {score}% of the selected role's requirements!</p>
    </div>
  );
}

export default ScoreDisplay; 