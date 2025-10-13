import { test, expect } from '@playwright/test';

/**
 * Backend API Validation Tests
 * Direct API testing to verify backend functionality
 */

test.describe('Backend API Validation', () => {
  const API_BASE = 'http://localhost:8007';

  test('Health check endpoint works', async ({ request }) => {
    const response = await request.get(`${API_BASE}/health`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('status', 'healthy');
    expect(data).toHaveProperty('service', 'enterprise-dashboard-api');
  });

  test('Dashboard metrics endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/dashboard/metrics`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('total_models');
    expect(data).toHaveProperty('active_jobs');
    expect(data).toHaveProperty('total_attacks');
    expect(data).toHaveProperty('detection_rate');
    expect(data).toHaveProperty('system_health');
    expect(data).toHaveProperty('last_updated');
    
    // Validate data types
    expect(typeof data.total_models).toBe('number');
    expect(typeof data.active_jobs).toBe('number');
    expect(typeof data.total_attacks).toBe('number');
    expect(typeof data.detection_rate).toBe('number');
    expect(typeof data.system_health).toBe('number');
  });

  test('Models available endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/models/available`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('models');
    expect(data).toHaveProperty('available_models');
    expect(data).toHaveProperty('mlflow_models');
    
    // Validate models structure
    expect(typeof data.models).toBe('object');
    expect(Array.isArray(data.available_models)).toBe(true);
    expect(Array.isArray(data.mlflow_models)).toBe(true);
  });

  test('Red team results endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/red-team/results`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('test_id');
    expect(data).toHaveProperty('timestamp');
    expect(data).toHaveProperty('total_attacks');
    expect(data).toHaveProperty('vulnerabilities_found');
    expect(data).toHaveProperty('detection_rate');
    expect(data).toHaveProperty('overall_status');
    expect(data).toHaveProperty('attacks');
    expect(data).toHaveProperty('results');
    
    // Validate data types
    expect(typeof data.test_id).toBe('string');
    expect(typeof data.total_attacks).toBe('number');
    expect(typeof data.vulnerabilities_found).toBe('number');
    expect(typeof data.detection_rate).toBe('number');
    expect(Array.isArray(data.attacks)).toBe(true);
    expect(Array.isArray(data.results)).toBe(true);
  });

  test('Analytics summary endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/analytics/summary?days=7`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('summary');
    expect(Array.isArray(data.summary)).toBe(true);
    
    if (data.summary.length > 0) {
      const summary = data.summary[0];
      expect(summary).toHaveProperty('model_name');
      expect(summary).toHaveProperty('model_type');
      expect(summary).toHaveProperty('total_tests');
      expect(summary).toHaveProperty('avg_detection_rate');
      expect(summary).toHaveProperty('avg_attacks');
      expect(summary).toHaveProperty('avg_vulnerabilities');
    }
  });

  test('Analytics trends endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/analytics/trends?days=7`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('trends');
    expect(Array.isArray(data.trends)).toBe(true);
    
    if (data.trends.length > 0) {
      const trend = data.trends[0];
      expect(trend).toHaveProperty('test_date');
      expect(trend).toHaveProperty('model_name');
      expect(trend).toHaveProperty('model_type');
      expect(trend).toHaveProperty('avg_detection_rate');
      expect(trend).toHaveProperty('test_count');
    }
  });

  test('Training jobs endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/training/jobs`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
  });

  test('MLflow experiments endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_BASE}/mlflow/experiments`);
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
  });

  test('Model prediction endpoint works correctly', async ({ request }) => {
    const response = await request.post(`${API_BASE}/models/predict`, {
      data: {
        text: 'Test prompt injection attack',
        model_name: 'distilbert_trained'
      }
    });
    
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('text');
    expect(data).toHaveProperty('prediction');
    expect(data).toHaveProperty('confidence');
    expect(data).toHaveProperty('probabilities');
    expect(data).toHaveProperty('model_predictions');
    expect(data).toHaveProperty('processing_time_ms');
    expect(data).toHaveProperty('timestamp');
  });

  test('Red team test endpoint works correctly', async ({ request }) => {
    const response = await request.post(`${API_BASE}/red-team/test`, {
      data: {
        model_name: 'distilbert_trained',
        test_count: 5,
        attack_categories: ['prompt_injection', 'jailbreak']
      }
    });
    
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('test_id');
    expect(data).toHaveProperty('status');
    expect(data).toHaveProperty('message');
  });

  test('Error handling for invalid endpoints', async ({ request }) => {
    const response = await request.get(`${API_BASE}/invalid-endpoint`);
    expect(response.status()).toBe(404);
    
    const data = await response.json();
    expect(data).toHaveProperty('detail');
  });

  test('Error handling for invalid request data', async ({ request }) => {
    const response = await request.post(`${API_BASE}/models/predict`, {
      data: {
        // Missing required 'text' field
        model_name: 'distilbert_trained'
      }
    });
    
    expect(response.status()).toBe(422);
    
    const data = await response.json();
    expect(data).toHaveProperty('detail');
  });

  test('CORS headers are properly set', async ({ request }) => {
    const response = await request.get(`${API_BASE}/health`);
    expect(response.status()).toBe(200);
    
    const headers = response.headers();
    expect(headers['access-control-allow-origin']).toBeDefined();
    expect(headers['access-control-allow-methods']).toBeDefined();
    expect(headers['access-control-allow-headers']).toBeDefined();
  });

  test('Response times are acceptable', async ({ request }) => {
    const startTime = Date.now();
    const response = await request.get(`${API_BASE}/dashboard/metrics`);
    const endTime = Date.now();
    
    expect(response.status()).toBe(200);
    expect(endTime - startTime).toBeLessThan(5000); // Should respond within 5 seconds
  });

  test('Concurrent requests are handled properly', async ({ request }) => {
    const promises = Array.from({ length: 10 }, () => 
      request.get(`${API_BASE}/health`)
    );
    
    const responses = await Promise.all(promises);
    
    responses.forEach(response => {
      expect(response.status()).toBe(200);
    });
  });
});
