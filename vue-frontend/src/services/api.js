import axios from 'axios';
import authService from './authService';

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
    'X-API-Key': config.apiKey  // Make sure the API key is included
  }
});

// Add request interceptor to include authorization token
apiClient.interceptors.request.use(
  config => {
    // Add authentication header if available
    const authHeader = authService.getAuthHeader();
    if (authHeader.Authorization) {
      config.headers.Authorization = authHeader.Authorization;
    }
    
    // Make sure the API key is always included
    if (window.APP_CONFIG && window.APP_CONFIG.apiKey) {
      config.headers['X-API-Key'] = window.APP_CONFIG.apiKey;
    }
    
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response && error.response.status === 401) {
      // If we get an unauthorized error, log the user out
      authService.logout();
      // Redirect to login page
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Ensure proper error handling
apiClient.interceptors.response.use(
  response => response,
  error => {
    // Ignore canceled request errors
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }
    
    // Handle other errors
    console.error('API error:', error);
    return Promise.reject(error);
  }
);

export default apiClient;