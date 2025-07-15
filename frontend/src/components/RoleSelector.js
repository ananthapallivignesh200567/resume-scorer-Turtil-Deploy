import React from 'react';
import './RoleSelector.css';

function RoleSelector({ roles, selectedRole, onSelectRole }) {
  return (
    <div className="role-selector">
      <label htmlFor="role-dropdown">Select Job Role:</label>
      <select
        id="role-dropdown"
        value={selectedRole}
        onChange={e => onSelectRole(e.target.value)}
      >
        <option value="">-- Choose a role --</option>
        {roles.map(role => (
          <option key={role} value={role}>{role}</option>
        ))}
      </select>
    </div>
  );
}

export default RoleSelector; 