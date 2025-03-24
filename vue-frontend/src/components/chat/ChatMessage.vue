<template>
  <div :class="['message', message.role]">
    <div class="message-content">
      <div v-if="message.role === 'assistant'" class="avatar">🤖</div>
      <div v-else class="avatar">👤</div>
      
      <div class="content">
        <div v-if="message.error" class="error-message">
          {{ message.content }}
        </div>
        <div v-else v-html="formattedContent"></div>
        
        <!-- Sources -->
        <div v-if="message.sources && message.sources.length" class="sources">
          <h4>Sources:</h4>
          <ul>
            <li v-for="(source, index) in message.sources" :key="index">
              {{ formatCitation(source) }}
            </li>
          </ul>
        </div>
        
        <!-- Feedback buttons (only for assistant messages) -->
        <div v-if="message.role === 'assistant' && !message.error && !feedbackSubmitted" class="feedback">
          <button 
            class="feedback-btn positive" 
            @click="submitFeedback('positive')"
            :disabled="feedbackSubmitted"
          >
            👍 Helpful
          </button>
          <button 
            class="feedback-btn negative" 
            @click="showFeedbackForm = true"
            :disabled="feedbackSubmitted"
          >
            👎 Not Helpful
          </button>
        </div>

        <div v-if="feedbackSubmitted" class="feedback-confirmation">
          Thank you for your feedback!
        </div>        
        
        <!-- Detailed feedback form -->
        <div v-if="showFeedbackForm && !feedbackSubmitted" class="feedback-form">
          <textarea 
            v-model="feedbackText" 
            placeholder="What was wrong with this response?"
            :disabled="feedbackSubmitted"
          ></textarea>
          <div class="form-actions">
            <button @click="submitFeedback('negative')" :disabled="feedbackSubmitted">Submit</button>
            <button @click="showFeedbackForm = false" :disabled="feedbackSubmitted">Cancel</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useChatStore } from '../../stores/chat';
import { marked } from 'marked';

const props = defineProps({
  message: {
    type: Object,
    required: true
  }
});

const chatStore = useChatStore();
const showFeedbackForm = ref(false);
const feedbackText = ref('');

const feedbackSubmitted = ref(false);

const formattedContent = computed(() => {
  return marked(props.message.content);
});

function formatCitation(source) {
  // Extract filename and chunkId
  const filename = source.filename || 'Unknown';
  const chunkId = source.chunkId || 'Unknown';
  
  // If we have metadata, use it to create a richer citation
  if (source.metadata) {
    const metadata = source.metadata;
    let citation = metadata.title ? `${metadata.title}` : filename;
    
    if (metadata.itemType) {
      citation += ` [${metadata.itemType}]`;
    }
    
    // Add authors if available
    if (metadata.creators && metadata.creators.length > 0) {
      const authors = metadata.creators
        .filter(c => c.creatorType === 'author')
        .map(a => `${a.lastName}, ${a.firstName}`);
      
      if (authors.length > 0) {
        citation += ` by ${authors.join(', ')}`;
      }
    }
    
    // Add date if available
    if (metadata.date) {
      citation += ` (${metadata.date})`;
    }
    
    // Add page if available
    if (source.page) {
      citation += `, Page ${source.page}`;
    }
    
    return citation;
  }
  
  // Simple citation without metadata
  return `${filename} (Chunk ${chunkId})`;
}

async function submitFeedback(rating) {
  try {
    // Get the original request ID from the message if available
    // or use the message ID as a fallback
    const originalRequestId = props.message.requestId || props.message.id;
    
    await chatStore.submitFeedback({
      originalRequestId: originalRequestId,
      messageId: props.message.id, 
      rating: rating,
      feedbackText: feedbackText.value
    });
    
    // Set feedbackSubmitted to true to disable and hide the feedback controls
    feedbackSubmitted.value = true;
    
    // Hide the feedback form
    showFeedbackForm.value = false;
    
    // Reset feedback text
    feedbackText.value = '';
  } catch (error) {
    console.error('Failed to submit feedback:', error);
    // Show an error message briefly
    alert('Failed to submit feedback. Please try again.');
  }
}
</script>

<style scoped>
.message {
  margin-bottom: 16px;
  display: flex;
}

.message-content {
  display: flex;
  max-width: 80%;
}

.assistant .message-content {
  background-color: #f0f7ff;
  border-radius: 12px;
  padding: 12px;
}

.user .message-content {
  margin-left: auto;
  background-color: #e6f7e6;
  border-radius: 12px;
  padding: 12px;
}

.avatar {
  font-size: 24px;
  margin-right: 12px;
  align-self: flex-start;
}

.sources {
  margin-top: 16px;
  font-size: 0.85rem;
  color: #555;
}

.sources ul {
  margin: 0;
  padding-left: 16px;
}

.feedback {
  margin-top: 12px;
  display: flex;
  gap: 8px;
}

.feedback-btn {
  background: none;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color 0.2s;
}

.feedback-btn.positive:hover:not(:disabled) {
  background-color: #e6f7e6;
}

.feedback-btn.negative:hover:not(:disabled) {
  background-color: #fff0f0;
}

.feedback-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.feedback-form {
  margin-top: 12px;
}

.feedback-form textarea {
  width: 100%;
  min-height: 80px;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 8px;
  margin-bottom: 8px;
}

.feedback-form textarea:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.form-actions {
  display: flex;
  gap: 8px;
}

.form-actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-message {
  color: #d32f2f;
}

.feedback-confirmation {
  margin-top: 12px;
  padding: 8px;
  background-color: #e6f7e6;
  border-radius: 4px;
  font-size: 0.9rem;
  color: #2e7d32;
  text-align: center;
}
</style>