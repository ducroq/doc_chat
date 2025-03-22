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
        // Add user message to UI
        this.messages.push({
          role: 'user',
          content,
          id: Date.now().toString(),
        });
        
        // Add to conversation history for context
        this.conversationHistory.push({
          role: 'user',
          content,
          timestamp: Date.now()
        });
        
        // Send to API
        const response = await chatService.sendMessage(content, this.conversationHistory);
        
        // Add response to messages
        this.messages.push({
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          id: Date.now().toString(),
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
    
    async submitFeedback(messageId, rating, feedbackText = null) {
      return chatService.submitFeedback({
        message_id: messageId,
        request_id: Date.now().toString(),
        rating,
        feedback_text: feedbackText,
        timestamp: new Date().toISOString()
      });
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