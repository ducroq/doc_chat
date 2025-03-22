import api from './api';

export default {
  async login(username, password) {
    try {
      const response = await api.post('/login', { username, password });
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  logout() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('username');
  }
}