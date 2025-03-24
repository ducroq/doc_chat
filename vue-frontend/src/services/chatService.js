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
  async submitFeedback(feedbackData) {
    try {
      console.log('Sending feedback data to API:', feedbackData);
  
      // Validate required fields before sending
      if (!feedbackData.request_id || !feedbackData.message_id || !feedbackData.rating || !feedbackData.timestamp) {
        throw new Error('Missing required feedback fields');
      }
      
      // Ensure rating is in correct format
      if (feedbackData.rating !== 'positive' && feedbackData.rating !== 'negative') {
        throw new Error('Rating must be "positive" or "negative"');
      }
  
      // Create a payload with only the fields the API expects
      const apiPayload = {
        request_id: String(feedbackData.request_id),
        message_id: String(feedbackData.message_id),
        rating: feedbackData.rating,
        feedback_text: feedbackData.feedback_text || null,
        categories: Array.isArray(feedbackData.categories) ? feedbackData.categories : [],
        timestamp: feedbackData.timestamp
      };
  
      const response = await api.post('/feedback', apiPayload);
      return response.data;
    } catch (error) {
      console.error('Error submitting feedback:', error);
      console.error('Error response:', error.response?.data);
      throw error;
    }
  }
}

