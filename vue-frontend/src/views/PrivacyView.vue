<template>
    <div class="privacy-container">
      <h1>Privacy Notice</h1>
      <div class="privacy-content">
        <p>When enabled, this system may log chat interactions for research and service improvement.</p>
        
        <h2>What we collect:</h2>
        <ul>
          <li>Questions asked to the system</li>
          <li>Responses provided</li>
          <li>Document references used</li>
          <li>Anonymized session identifiers</li>
        </ul>
        
        <h2>Data Protection:</h2>
        <ul>
          <li>All identifiers are anonymized</li>
          <li>Logs are automatically deleted after 30 days</li>
          <li>Data is stored securely within the EU</li>
          <li>You can request deletion of your data</li>
        </ul>
        
        <button @click="goBack">Back to Chat</button>
      </div>
    </div>
  </template>
  
  <script setup>
  import { useRouter } from 'vue-router';
  
  const router = useRouter();
  
  function goBack() {
    router.push('/');
  }
  </script>
  
  <style scoped>
  .privacy-container {
    padding: 2rem;
    max-width: 800px;
    margin: 0 auto;
  }
  
  .privacy-content {
    background-color: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  h1 {
    margin-bottom: 2rem;
  }
  
  h2 {
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
  }
  
  ul {
    margin-bottom: 1.5rem;
  }
  
  button {
    margin-top: 2rem;
    padding: 0.5rem 1rem;
    background-color: #4a6cf7;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  </style>