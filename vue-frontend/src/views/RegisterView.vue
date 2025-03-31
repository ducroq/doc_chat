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
  import { ref } from 'vue';
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
  
  async function register() {
    // Reset error message
    error.value = '';
    
    // Validate form
    if (password.value !== confirmPassword.value) {
      error.value = 'Passwords do not match';
      return;
    }
    
    loading.value = true;
    
    try {
      // Send registration request to API
      await api.post('/register', {
        username: username.value,
        password: password.value,
        full_name: fullName.value || undefined,
        email: email.value || undefined
      });
      
      // Registration successful, redirect to login
      router.push('/login?registered=true');
    } catch (err) {
      console.error('Registration error:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        error.value = err.response.data.detail;
      } else {
        error.value = 'Registration failed. Please try again.';
      }
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