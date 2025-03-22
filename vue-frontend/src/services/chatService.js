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
  }
}
