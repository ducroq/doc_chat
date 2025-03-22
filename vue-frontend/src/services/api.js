import axios from 'axios';

// Get config from window.APP_CONFIG (injected at runtime)
const config = window.APP_CONFIG || {
  apiUrl: '/api',  // Use relative path for browser requests
  apiKey: ''
};

const apiClient = axios.create({
  baseURL: config.apiUrl,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': config.apiKey
  }
});

export default apiClient;