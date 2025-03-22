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
        <div v-if="message.role === 'assistant' && !message.error" class="feedback">
          <button 
            class="feedback-btn positive" 
            @click="submitFeedback('positive')"
          >
            👍 Helpful
          </button>
          <button 
            class="feedback-btn negative" 
            @click="showFeedbackForm = true"
          >
            👎 Not Helpful
          </button>
        </div>
        
        <!-- Detailed feedback form -->
        <div v-if="showFeedbackForm" class="feedback-form">
          <textarea 
            v-model="feedbackText" 
            placeholder="What was wrong with this response?"
          ></textarea>
          <div class="form-actions">
            <button @click="submitFeedback('negative')">Submit</button>
            <button @click="showFeedbackForm = false">Cancel</button>
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
    await chatStore.submitFeedback(props.message.id, rating, feedbackText.value);
    showFeedbackForm.value = false;
    feedbackText.value = '';
    // Show success notification
  } catch (error) {
    console.error('Failed to submit feedback:', error);
    // Show error notification
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

.feedback-btn.positive:hover {
  background-color: #e6f7e6;
}

.feedback-btn.negative:hover {
  background-color: #fff0f0;
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

.form-actions {
  display: flex;
  gap: 8px;
}

.error-message {
  color: #d32f2f;
}
</style>