import { defineStore } from 'pinia';
import chatService from '../services/chatService';

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [],
    conversationHistory: [],
    isLoading: false,
    error: null,
    systemStatus: {
      api: 'unknown',
      weaviate: 'unknown',
      mistral_api: 'unknown'
    }
  }),
  
  actions: {
    async sendMessage(content) {
      this.isLoading = true;
      this.error = null;
      
      try {
        // Generate a unique request ID for this conversation
        const requestId = Date.now().toString();
        
        // Add user message to UI
        this.messages.push({
          role: 'user',
          content,
          id: Date.now().toString(),
          requestId: requestId
        });
        
        // Add to conversation history for context
        this.conversationHistory.push({
          role: 'user',
          content,
          timestamp: Date.now()
        });
        
        // Send to API
        const response = await chatService.sendMessage(content, this.conversationHistory);
        
        // Add response to messages - include the same requestId for tracking
        this.messages.push({
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          id: Date.now().toString(),
          requestId: requestId  // Use the same requestId to link messages
        });
        
        // Add to conversation history
        this.conversationHistory.push({
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          timestamp: Date.now()
        });
        
        return response;
      } catch (error) {
        this.error = error.message || 'Failed to send message';
        // Add error message to chat
        this.messages.push({
          role: 'assistant',
          content: `Error: ${this.error}`,
          error: true,
          id: Date.now().toString()
        });
        throw error;
      } finally {
        this.isLoading = false;
      }
    },

    async submitFeedback(feedbackParams) {
      try {
        // Create a feedback object that exactly matches what the API expects
        const feedbackData = {
          request_id: Date.now().toString(), // A unique ID for this feedback submission
          message_id: feedbackParams.messageId,
          rating: feedbackParams.rating, // Must be "positive" or "negative"
          feedback_text: feedbackParams.feedbackText || null,
          categories: [], // Optional categories if implemented
          timestamp: new Date().toISOString() // Must be ISO format
        };
    
        console.log('Submitting feedback:', feedbackData);
        
        // Send the feedback to the API
        const response = await chatService.submitFeedback(feedbackData);
        console.log('Feedback submitted successfully:', response);
        return response;
      } catch (error) {
        console.error('Failed to submit feedback:', error);
        throw error;
      }
    },
    
    async checkSystemStatus() {
      try {
        const response = await chatService.getSystemStatus();
        this.systemStatus = response;
        return response;
      } catch (error) {
        console.error('Failed to check system status:', error);
        return null;
      }
    },
    
    clearChat() {
      this.messages = [];
      this.conversationHistory = [];
    }
  }
});