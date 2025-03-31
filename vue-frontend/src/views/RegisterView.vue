<template>
    <div class="register-container">
      <div class="register-card">
        <h1>🇪🇺 Document Chat</h1>
        <h2>Create Account</h2>
        
        <form @submit.prevent="register">
          <div class="form-group">
            <label for="username">Username</label>
            <input 
              type="text" 
              id="username" 
              v-model="username" 
              required
              placeholder="Choose a username"
            />
          </div>
          
          <div class="form-group">
            <label for="password">Password</label>
            <input 
              type="password" 
              id="password" 
              v-model="password" 
              required
              placeholder="Create a strong password"
            />
            <small class="password-requirements">
              Password must be at least 8 characters with uppercase, lowercase, 
              numbers, and special characters.
            </small>
          </div>
          
          <div class="form-group">
            <label for="confirmPassword">Confirm Password</label>
            <input 
              type="password" 
              id="confirmPassword" 
              v-model="confirmPassword" 
              required
              placeholder="Confirm your password"
            />
          </div>
          
          <div class="form-group">
            <label for="fullName">Full Name (Optional)</label>
            <input 
              type="text" 
              id="fullName" 
              v-model="fullName" 
              placeholder="Your full name"
            />
          </div>
          
          <div class="form-group">
            <label for="email">Email (Optional)</label>
            <input 
              type="email" 
              id="email" 
              v-model="email" 
              placeholder="Your email address"
            />
          </div>
          <div class="form-group captcha">              
            <label>Please solve this math problem: {{ captchaQuestion }}</label>
            <input 
              type="number" 
              v-model="captchaAnswer" 
              required
              placeholder="Enter answer"
            />
            <button type="button" @click="refreshCaptcha" class="refresh-captcha">
              Refresh
            </button>
          </div>          
          
          <p v-if="error" class="error-message">{{ error }}</p>
          
          <button type="submit" :disabled="loading">
            {{ loading ? 'Creating Account...' : 'Create Account' }}
          </button>
          
          <p class="login-link">
            Already have an account? <router-link to="/login">Log in</router-link>
          </p>
        </form>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import { useRouter } from 'vue-router';
  import api from '../services/api';
  
  const router = useRouter();
  const username = ref('');
  const password = ref('');
  const confirmPassword = ref('');
  const fullName = ref('');
  const email = ref('');
  const error = ref('');
  const loading = ref(false);

  const captchaQuestion = ref('');
  const captchaAnswer = ref('');
  const captchaHash = ref('');
  const captchaTimestamp = ref(0);
  
  async function fetchCaptcha() {
    try {
      captchaAnswer.value = ''; // Clear the previous answer
      const response = await api.get('/captcha');
      captchaQuestion.value = response.data.question;
      captchaHash.value = response.data.hash;
      captchaTimestamp.value = response.data.timestamp;
    } catch (err) {
      console.error('Failed to load CAPTCHA:', err);
      captchaQuestion.value = 'Error loading math problem. Please refresh the page.';
    }
  }

  function refreshCaptcha() {
    captchaAnswer.value = '';
    fetchCaptcha();
  }

  // Load CAPTCHA when the component is mounted
  onMounted(() => {
    fetchCaptcha();
  });

  async function register() {
    if (password.value !== confirmPassword.value) {
      error.value = 'Passwords do not match';
      return;
    }

    loading.value = true;
    
    try {
      // Create FormData object
      const formData = new FormData();
      formData.append('username', username.value);
      formData.append('password', password.value);
      if (fullName.value) formData.append('full_name', fullName.value);
      if (email.value) formData.append('email', email.value);
      formData.append('captcha_answer', captchaAnswer.value);
      formData.append('captcha_hash', captchaHash.value);
      formData.append('captcha_timestamp', captchaTimestamp.value);
      
      // Send as FormData
      await api.post('/register', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      router.push('/login?registered=true');
    } catch (err) {
      console.error('Registration error:', err);
    
      if (err.response) {
        if (err.response.status === 400 && err.response.data && 
            err.response.data.detail && err.response.data.detail.includes('CAPTCHA')) {
          // For CAPTCHA errors, show a friendlier message
          error.value = 'The answer to the math problem was incorrect. Please try again.';
        } else if (err.response.data && err.response.data.detail) {
          // Other validation errors
          error.value = err.response.data.detail;
        } else {
          // Generic error with status code
          error.value = `Error ${err.response.status}: Please try again.`;
        }
      } else {
        // Network or other errors
        error.value = 'Registration failed. Please try again later.';
      }
      
      // Refresh CAPTCHA if there was an error
      fetchCaptcha();
    } finally {
      loading.value = false;
    }
  }
  </script>
  
  <style scoped>
  /* Reuse styles from LoginView.vue with some adjustments */
  .register-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background-color: #f5f7fb;
    padding: 20px;
  }
  
  .register-card {
    width: 100%;
    max-width: 500px;
    padding: 2rem;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  /* Rest of your styles from LoginView.vue, plus: */
  .password-requirements {
    display: block;
    margin-top: 4px;
    color: #666;
    font-size: 0.8rem;
  }
  
  .login-link {
    margin-top: 1rem;
    text-align: center;
    font-size: 0.9rem;
  }
  
  .error-message {
    color: #e53e3e;
    margin-bottom: 1rem;
  }
  </style>