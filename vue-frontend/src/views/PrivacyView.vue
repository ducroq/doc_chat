<template>
  <div class="privacy-container">
    <h1>Chat Logging Privacy Notice</h1>
    
    <div class="privacy-content">
      <div class="highlight">
        <p><strong>Summary:</strong> When enabled, this system logs anonymized chat interactions for research purposes. Logs are automatically deleted after 30 days, and you can request earlier deletion of your data at any time.</p>
      </div>
      
      <h2>Introduction</h2>
      <p>The Document Chat System may collect and process interaction data for research and service improvement purposes. This privacy notice explains how we handle this data in compliance with GDPR and other applicable privacy regulations.</p>
      
      <h2>Data Collection</h2>
      <p>When enabled, the system logs the following information:</p>
      <ul>
        <li><strong>Date and time</strong> of interactions</li>
        <li><strong>Questions asked</strong> to the system</li>
        <li><strong>Responses provided</strong> by the system</li>
        <li><strong>Document references</strong> used to generate answers</li>
        <li><strong>Anonymized session identifiers</strong> to link related interactions</li>
        <li><strong>System performance metrics</strong></li>
      </ul>

      <h2>Feedback Collection</h2>
      <p>When enabled, the system also collects feedback on responses:</p>
      <ul>
        <li><strong>Rating information</strong> (whether a response was helpful)</li>
        <li><strong>Detailed feedback</strong> you choose to provide</li>
        <li><strong>Categories of issues</strong> you identify with responses</li>
        <li><strong>Anonymized session identifiers</strong> to link feedback to interactions</li>
      </ul>

      <p>This feedback data is processed and stored with the same privacy protections, anonymization, and retention policies as other interaction data.</p>
      
      <h2>Legal Basis for Processing</h2>
      <p>We process this data based on:</p>
      <ul>
        <li>Our legitimate interest in improving the system's accuracy and performance</li>
        <li>Your consent, which you may withdraw at any time</li>
        <li>Research purposes (statistical and scientific analysis)</li>
      </ul>
      
      <h2>How We Use This Data</h2>
      <p>The collected data is used to:</p>
      <ul>
        <li>Analyze common question patterns</li>
        <li>Improve answer accuracy and relevance</li>
        <li>Identify missing information in our document collection</li>
        <li>Fix technical issues and optimize system performance</li>
        <li>Conduct research on information retrieval and question-answering systems</li>
        <li>Improve the quality and accuracy of responses based on your feedback</li>
        <li>Address common issues identified through feedback analysis</li>
      </ul>
      
      <h2>Data Retention</h2>
      <ul>
        <li>Chat logs are retained for a maximum of 30 days by default</li>
        <li>After this period, logs are automatically deleted</li>
        <li>You may request earlier deletion of your data at any time</li>
      </ul>
      
      <h2>Data Security and Access</h2>
      <ul>
        <li>All logs are stored securely with access restricted to authorized administrators</li>
        <li>Logs are stored locally on the system's secured servers</li>
        <li>We implement appropriate technical and organizational measures to protect your data</li>
      </ul>
      
      <h2>Anonymization</h2>
      <p>By default, when logging is enabled:</p>
      <ul>
        <li>User identifiers are anonymized through cryptographic hashing</li>
        <li>IP addresses are not stored or are anonymized</li>
        <li>Personal information in queries may still be recorded; avoid including sensitive data in your questions</li>
      </ul>
      
      <div class="highlight">
        <p><strong>Important:</strong> The system may record the content of your questions. Please do not include personal data, sensitive information, or confidential details in your queries.</p>
      </div>
      
      <h2>Your Rights</h2>
      <p>Under GDPR and similar regulations, you have the right to:</p>
      <ul>
        <li><strong>Access</strong> the data we hold about you</li>
        <li>Request <strong>correction</strong> of inaccurate data</li>
        <li>Request <strong>deletion</strong> of your data</li>
        <li><strong>Object to</strong> or <strong>restrict</strong> the processing of your data</li>
        <li><strong>Withdraw consent</strong> for data collection at any time</li>
      </ul>
      
      <h2>How to Exercise Your Rights</h2>
      <p>To exercise any of these rights, please contact the system administrator. We will respond to your request within 30 days.</p>
      
      <h2>Changes to This Privacy Notice</h2>
      <p>We may update this privacy notice from time to time. Any changes will be posted with a revised effective date.</p>
      
      <div class="footer">
        <p>Effective Date: March 9, 2025</p>
        <p>Document version: 1.0</p>
      </div>
      
      <button @click="goBack" class="back-button">Back to Chat</button>
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
  color: #2c3e50;
  border-bottom: 2px solid #3498db;
  padding-bottom: 10px;
  margin-bottom: 2rem;
}

h2 {
  color: #3498db;
  margin-top: 30px;
  border-left: 4px solid #3498db;
  padding-left: 10px;
  margin-bottom: 0.5rem;
}

ul {
  margin-bottom: 1.5rem;
}

.highlight {
  background-color: #f8f9fa;
  border-left: 4px solid #4caf50;
  padding: 10px 15px;
  margin: 20px 0;
}

.footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #eee;
  font-size: 0.9em;
  color: #777;
}

.back-button {
  margin-top: 2rem;
  padding: 0.75rem 1.5rem;
  background-color: #4a6cf7;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.2s;
}

.back-button:hover {
  background-color: #3a5cd7;
}
</style>