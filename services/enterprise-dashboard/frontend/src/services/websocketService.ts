import React from 'react';

/**
 * WebSocket Service for Real-time Dashboard Updates
 * Handles WebSocket connections and message broadcasting
 */

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
}

export interface WebSocketSubscription {
  topic: string;
  callback: (message: WebSocketMessage) => void;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 30000; // Max 30 seconds
  private subscriptions: Map<string, Set<(message: WebSocketMessage) => void>> = new Map();
  private isConnecting = false;
  private shouldReconnect = true;
  private pingInterval: NodeJS.Timeout | null = null;
  private pongTimeout: NodeJS.Timeout | null = null;

  constructor() {
    this.connect();
  }

  private connect(): void {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isConnecting = true;
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8007/ws';
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.handleReconnect();
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('✅ WebSocket connected');
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.startPingInterval();
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      this.isConnecting = false;
      this.stopPingInterval();
      
      if (this.shouldReconnect && !event.wasClean) {
        this.handleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.isConnecting = false;
    };
  }

  private handleMessage(message: WebSocketMessage): void {
    console.log('📨 WebSocket message received:', message);

    // Handle pong responses
    if (message.type === 'pong') {
      if (this.pongTimeout) {
        clearTimeout(this.pongTimeout);
        this.pongTimeout = null;
      }
      return;
    }

    // Broadcast to all subscribers
    this.subscriptions.forEach((callbacks) => {
      callbacks.forEach((callback) => {
        try {
          callback(message);
        } catch (error) {
          console.error('Error in WebSocket callback:', error);
        }
      });
    });
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startPingInterval(): void {
    this.stopPingInterval();
    this.pingInterval = setInterval(() => {
      this.ping();
    }, 30000); // Ping every 30 seconds
  }

  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    if (this.pongTimeout) {
      clearTimeout(this.pongTimeout);
      this.pongTimeout = null;
    }
  }

  private ping(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: 'ping',
        timestamp: new Date().toISOString()
      });

      // Set timeout for pong response
      this.pongTimeout = setTimeout(() => {
        console.warn('Pong timeout, reconnecting...');
        this.ws?.close();
      }, 5000);
    }
  }

  public send(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  public subscribe(topic: string, callback: (message: WebSocketMessage) => void): () => void {
    if (!this.subscriptions.has(topic)) {
      this.subscriptions.set(topic, new Set());
    }
    
    this.subscriptions.get(topic)!.add(callback);

    // Send subscription message to server
    this.send({
      type: 'subscribe',
      topic: topic,
      timestamp: new Date().toISOString()
    });

    // Return unsubscribe function
    return () => {
      this.unsubscribe(topic, callback);
    };
  }

  public unsubscribe(topic: string, callback: (message: WebSocketMessage) => void): void {
    const callbacks = this.subscriptions.get(topic);
    if (callbacks) {
      callbacks.delete(callback);
      if (callbacks.size === 0) {
        this.subscriptions.delete(topic);
        // Send unsubscribe message to server
        this.send({
          type: 'unsubscribe',
          topic: topic,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  public getConnectionState(): number {
    return this.ws ? this.ws.readyState : WebSocket.CLOSED;
  }

  public isConnected(): boolean {
    return this.ws ? this.ws.readyState === WebSocket.OPEN : false;
  }

  public disconnect(): void {
    this.shouldReconnect = false;
    this.stopPingInterval();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  public reconnect(): void {
    this.shouldReconnect = true;
    this.reconnectAttempts = 0;
    this.disconnect();
    this.connect();
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

// React hook for using WebSocket
export const useWebSocket = () => {
  const [isConnected, setIsConnected] = React.useState(webSocketService.isConnected());
  const [lastMessage, setLastMessage] = React.useState<WebSocketMessage | null>(null);

  React.useEffect(() => {
    const unsubscribe = webSocketService.subscribe('all', (message) => {
      setLastMessage(message);
    });

    // Check connection status periodically
    const statusInterval = setInterval(() => {
      setIsConnected(webSocketService.isConnected());
    }, 1000);

    return () => {
      unsubscribe();
      clearInterval(statusInterval);
    };
  }, []);

  return {
    isConnected,
    lastMessage,
    subscribe: webSocketService.subscribe.bind(webSocketService),
    unsubscribe: webSocketService.unsubscribe.bind(webSocketService),
    send: webSocketService.send.bind(webSocketService),
    reconnect: webSocketService.reconnect.bind(webSocketService)
  };
};

export default webSocketService;