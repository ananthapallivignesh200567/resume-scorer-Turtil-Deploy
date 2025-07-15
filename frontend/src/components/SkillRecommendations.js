import React from 'react';
import './SkillRecommendations.css';

function SkillRecommendations({ recommended, matched, missing }) {
  return (
    <div className="skill-recommendations">
      <div className="skill-section">
        <h3>Recommended Skills to Add</h3>
        <div className="skill-chips">
          {recommended.map(skill => (
            <span className="skill-chip recommended" key={skill}>⭐ {skill}</span>
          ))}
        </div>
      </div>
      <div className="skill-section">
        <h3>Skills You Have</h3>
        <div className="skill-chips">
          {matched.map(skill => (
            <span className="skill-chip matched" key={skill}>✔️ {skill}</span>
          ))}
        </div>
      </div>
      <div className="skill-section">
        <h3>Missing Skills</h3>
        <div className="skill-chips">
          {missing.map(skill => (
            <span className="skill-chip missing" key={skill}>⚠️ {skill}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default SkillRecommendations; 