import api from './api';

export default {
  async login(username, password) {
    try {
      const response = await api.post('/login', { username, password });
      
      // Store auth token and user info in localStorage
      localStorage.setItem('auth_token', response.data.access_token);
      localStorage.setItem('token_type', response.data.token_type);
      localStorage.setItem('username', response.data.username);
      localStorage.setItem('full_name', response.data.full_name || '');
      localStorage.setItem('isAuthenticated', 'true');
      
      return response.data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },
  
  logout() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_type');
    localStorage.removeItem('username');
    localStorage.removeItem('full_name');
    localStorage.removeItem('isAuthenticated');
  },
  
  getAuthHeader() {
    const token = localStorage.getItem('auth_token');
    const tokenType = localStorage.getItem('token_type') || 'Bearer';
    
    if (token) {
      return { Authorization: `${tokenType} ${token}` };
    }
    return {};
  },
  
  isAuthenticated() {
    return localStorage.getItem('isAuthenticated') === 'true';
  },
  
  getCurrentUser() {
    return {
      username: localStorage.getItem('username'),
      fullName: localStorage.getItem('full_name')
    };
  }
}