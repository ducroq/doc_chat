import api from './api';

export default {
  async sendMessage(message, conversationHistory) {
    try {
      const response = await api.post('/chat', {
        question: message,
        conversation_history: conversationHistory
      });
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },
  
  async submitFeedback(feedback) {
    return api.post('/feedback', feedback);
  },
  
  async getSystemStatus() {
    return api.get('/status');
  }
};