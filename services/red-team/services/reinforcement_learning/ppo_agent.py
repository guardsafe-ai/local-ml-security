"""
PPO (Proximal Policy Optimization) Agent for Attack Planning
Implements PPO algorithm for strategic attack planning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PPOAttackAgent:
    """
    PPO Agent for strategic attack planning
    Implements Proximal Policy Optimization for learning optimal attack strategies
    """
    
    def __init__(self, 
                 state_dim: int = 128,
                 action_dim: int = 64,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs for PPO update
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        # Neural networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.memory = ExperienceBuffer()
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_history = []
        
        logger.info(f"✅ PPO Attack Agent initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action based on current state
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get action distribution and value
                action_dist = self.actor(state_tensor)
                value = self.critic(state_tensor)
                
                if deterministic:
                    action = action_dist.mean
                else:
                    action = action_dist.sample()
                
                log_prob = action_dist.log_prob(action)
                
                # Convert to numpy
                action_np = action.cpu().numpy().flatten()
                log_prob_np = log_prob.cpu().numpy().flatten()[0]
                value_np = value.cpu().numpy().flatten()[0]
                
                return action_np, log_prob_np, value_np
                
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            # Return random action as fallback
            return np.random.randn(self.action_dim), 0.0, 0.0
    
    def store_transition(self, 
                        state: np.ndarray, 
                        action: np.ndarray, 
                        reward: float, 
                        next_state: np.ndarray, 
                        done: bool,
                        log_prob: float,
                        value: float) -> None:
        """
        Store transition in experience buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate
        """
        try:
            self.memory.store(state, action, reward, next_state, done, log_prob, value)
            
        except Exception as e:
            logger.error(f"Transition storage failed: {e}")
    
    def update(self) -> Dict[str, float]:
        """
        Update the agent using PPO algorithm
        
        Returns:
            Training metrics
        """
        try:
            if len(self.memory) < 32:  # Minimum batch size
                return {"error": "Insufficient data for update"}
            
            # Get batch data
            states, actions, rewards, next_states, dones, log_probs, values = self.memory.get_batch()
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs).to(self.device)
            old_values = torch.FloatTensor(values).to(self.device)
            
            # Calculate advantages and returns
            advantages, returns = self._calculate_advantages_returns(rewards, dones, old_values)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            actor_losses = []
            critic_losses = []
            
            for _ in range(self.k_epochs):
                # Actor update
                action_dist = self.actor(states)
                new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
                entropy = action_dist.entropy().sum(dim=-1)
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Critic update
                new_values = self.critic(states).squeeze()
                critic_loss = F.mse_loss(new_values, returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
            
            # Clear memory
            self.memory.clear()
            
            # Record training metrics
            metrics = {
                'actor_loss': np.mean(actor_losses),
                'critic_loss': np.mean(critic_losses),
                'avg_advantage': advantages.mean().item(),
                'avg_return': returns.mean().item(),
                'training_step': self.training_step
            }
            
            self.training_step += 1
            self.training_history.append(metrics)
            
            logger.debug(f"PPO update completed: actor_loss={metrics['actor_loss']:.4f}, critic_loss={metrics['critic_loss']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"PPO update failed: {e}")
            return {"error": str(e)}
    
    def _calculate_advantages_returns(self, 
                                    rewards: torch.Tensor, 
                                    dones: torch.Tensor, 
                                    values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate advantages and returns using GAE
        
        Args:
            rewards: Rewards tensor
            dones: Done flags tensor
            values: Value estimates tensor
            
        Returns:
            Tuple of (advantages, returns)
        """
        try:
            advantages = []
            returns = []
            
            # Calculate advantages using GAE
            advantage = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0 if dones[t] else values[t]
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                advantage = delta + self.gamma * 0.95 * advantage * (1 - dones[t])  # GAE with lambda=0.95
                advantages.insert(0, advantage)
            
            # Calculate returns
            for t in range(len(rewards)):
                if t == len(rewards) - 1:
                    returns.append(rewards[t])
                else:
                    return_val = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
                    returns.insert(0, return_val)
            
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            return advantages, returns
            
        except Exception as e:
            logger.error(f"Advantages/returns calculation failed: {e}")
            return torch.zeros_like(rewards), torch.zeros_like(rewards)
    
    def train_episode(self, 
                     environment: 'AttackEnvironment', 
                     max_steps: int = 100) -> Dict[str, Any]:
        """
        Train the agent for one episode
        
        Args:
            environment: Attack environment
            max_steps: Maximum steps per episode
            
        Returns:
            Episode metrics
        """
        try:
            state = environment.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.select_action(state)
                
                # Take action in environment
                next_state, reward, done, info = environment.step(action)
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done, log_prob, value)
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Record episode metrics
            episode_metrics = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'final_state': state.tolist() if isinstance(state, np.ndarray) else state
            }
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            logger.debug(f"Episode completed: reward={episode_reward:.2f}, length={episode_length}")
            return episode_metrics
            
        except Exception as e:
            logger.error(f"Episode training failed: {e}")
            return {"error": str(e)}
    
    def train(self, 
              environment: 'AttackEnvironment', 
              num_episodes: int = 1000,
              update_frequency: int = 10,
              max_steps_per_episode: int = 100) -> Dict[str, Any]:
        """
        Train the agent for multiple episodes
        
        Args:
            environment: Attack environment
            num_episodes: Number of episodes to train
            update_frequency: How often to update the agent
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Training results
        """
        try:
            logger.info(f"Starting training: {num_episodes} episodes, update_frequency={update_frequency}")
            
            training_results = {
                'episodes': [],
                'final_metrics': {},
                'training_completed': False
            }
            
            for episode in range(num_episodes):
                # Train one episode
                episode_metrics = self.train_episode(environment, max_steps_per_episode)
                episode_metrics['episode'] = episode
                training_results['episodes'].append(episode_metrics)
                
                # Update agent periodically
                if episode % update_frequency == 0 and episode > 0:
                    update_metrics = self.update()
                    episode_metrics['update_metrics'] = update_metrics
                
                # Log progress
                if episode % 100 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                    logger.info(f"Episode {episode}: avg_reward={avg_reward:.2f}")
            
            # Final update
            if len(self.memory) > 0:
                final_update = self.update()
                training_results['final_update'] = final_update
            
            # Calculate final metrics
            training_results['final_metrics'] = {
                'total_episodes': num_episodes,
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
                'min_reward': min(self.episode_rewards) if self.episode_rewards else 0,
                'training_steps': self.training_step
            }
            
            training_results['training_completed'] = True
            
            logger.info(f"Training completed: avg_reward={training_results['final_metrics']['avg_reward']:.2f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e), "training_completed": False}
    
    def evaluate(self, 
                environment: 'AttackEnvironment', 
                num_episodes: int = 10,
                max_steps_per_episode: int = 100) -> Dict[str, Any]:
        """
        Evaluate the agent
        
        Args:
            environment: Attack environment
            num_episodes: Number of episodes to evaluate
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Evaluation results
        """
        try:
            logger.info(f"Starting evaluation: {num_episodes} episodes")
            
            evaluation_rewards = []
            evaluation_lengths = []
            
            for episode in range(num_episodes):
                state = environment.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(max_steps_per_episode):
                    # Use deterministic action selection for evaluation
                    action, _, _ = self.select_action(state, deterministic=True)
                    
                    next_state, reward, done, info = environment.step(action)
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                evaluation_rewards.append(episode_reward)
                evaluation_lengths.append(episode_length)
            
            evaluation_results = {
                'num_episodes': num_episodes,
                'avg_reward': np.mean(evaluation_rewards),
                'std_reward': np.std(evaluation_rewards),
                'avg_length': np.mean(evaluation_lengths),
                'std_length': np.std(evaluation_lengths),
                'max_reward': max(evaluation_rewards),
                'min_reward': min(evaluation_rewards),
                'rewards': evaluation_rewards,
                'lengths': evaluation_lengths
            }
            
            logger.info(f"Evaluation completed: avg_reward={evaluation_results['avg_reward']:.2f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            model_state = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'training_step': self.training_step,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'training_history': self.training_history,
                'hyperparameters': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'hidden_dim': self.hidden_dim,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'eps_clip': self.eps_clip,
                    'k_epochs': self.k_epochs
                }
            }
            
            torch.save(model_state, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful
        """
        try:
            model_state = torch.load(filepath, map_location=self.device)
            
            self.actor.load_state_dict(model_state['actor_state_dict'])
            self.critic.load_state_dict(model_state['critic_state_dict'])
            self.actor_optimizer.load_state_dict(model_state['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(model_state['critic_optimizer_state_dict'])
            
            self.training_step = model_state.get('training_step', 0)
            self.episode_rewards = model_state.get('episode_rewards', [])
            self.episode_lengths = model_state.get('episode_lengths', [])
            self.training_history = model_state.get('training_history', [])
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Training statistics
        """
        try:
            stats = {
                'training_step': self.training_step,
                'total_episodes': len(self.episode_rewards),
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
                'min_reward': min(self.episode_rewards) if self.episode_rewards else 0,
                'recent_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                'training_history_length': len(self.training_history)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Training stats calculation failed: {e}")
            return {"error": str(e)}


class ActorNetwork(nn.Module):
    """Actor network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        
        # Create action distribution (assuming continuous actions)
        std = torch.ones_like(mean) * 0.1  # Fixed standard deviation
        dist = torch.distributions.Normal(mean, std)
        
        return dist


class CriticNetwork(nn.Module):
    """Critic network for PPO"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value


class ExperienceBuffer:
    """Experience buffer for storing transitions"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def get_batch(self):
        """Get all stored transitions"""
        if not self.buffer:
            return [], [], [], [], [], [], []
        
        states, actions, rewards, next_states, dones, log_probs, values = zip(*self.buffer)
        return list(states), list(actions), list(rewards), list(next_states), list(dones), list(log_probs), list(values)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
