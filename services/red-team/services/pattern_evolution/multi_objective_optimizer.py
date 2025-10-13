"""
Multi-Objective Optimizer
Advanced multi-objective optimization using NSGA-II and other algorithms
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import random
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class OptimizationAlgorithm(Enum):
    """Multi-objective optimization algorithms"""
    NSGA_II = "nsga_ii"
    NSGA_III = "nsga_iii"
    SPEA2 = "spea2"
    MOEA_D = "moea_d"
    IBEA = "ibea"


@dataclass
class ParetoFront:
    """Pareto front representation"""
    solutions: List[np.ndarray]
    objectives: List[np.ndarray]
    ranks: List[int]
    crowding_distances: List[float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization"""
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA_II
    population_size: int = 100
    max_generations: int = 1000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    reference_points: Optional[List[np.ndarray]] = None
    reference_directions: Optional[List[np.ndarray]] = None
    convergence_threshold: float = 1e-6
    max_stagnation: int = 50
    diversity_threshold: float = 0.1
    parallel_evaluation: bool = True
    custom_fitness_function: Optional[Callable] = None
    custom_crossover_function: Optional[Callable] = None
    custom_mutation_function: Optional[Callable] = None


class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimizer
    Implements various multi-objective optimization algorithms
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        """Initialize multi-objective optimizer"""
        self.config = config
        self.population: List[Dict[str, Any]] = []
        self.pareto_front: Optional[ParetoFront] = None
        self.generation = 0
        self.hypervolume_history: List[float] = []
        self.spread_history: List[float] = []
        self.converged = False
        
        logger.info("✅ Initialized Multi-Objective Optimizer")
    
    async def optimize(self, 
                      initial_population: Optional[List[Dict[str, Any]]] = None,
                      fitness_function: Optional[Callable] = None) -> ParetoFront:
        """
        Optimize using multi-objective algorithm
        """
        try:
            logger.info("Starting multi-objective optimization")
            
            # Initialize population
            if initial_population is None:
                await self._initialize_population()
            else:
                self.population = initial_population.copy()
            
            # Set fitness function
            if fitness_function is not None:
                self.config.custom_fitness_function = fitness_function
            
            # Evolution loop
            for generation in range(self.config.max_generations):
                self.generation = generation
                
                # Evaluate fitness
                await self._evaluate_fitness()
                
                # Update Pareto front
                await self._update_pareto_front()
                
                # Calculate metrics
                hypervolume = self._calculate_hypervolume()
                spread = self._calculate_spread()
                
                self.hypervolume_history.append(hypervolume)
                self.spread_history.append(spread)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Converged at generation {generation}")
                    break
                
                # Selection
                parents = await self._selection()
                
                # Crossover
                offspring = await self._crossover(parents)
                
                # Mutation
                await self._mutation(offspring)
                
                # Replacement
                await self._replacement(offspring)
                
                # Log progress
                if generation % 10 == 0:
                    logger.info(f"Generation {generation}: Hypervolume = {hypervolume:.6f}, Spread = {spread:.6f}")
            
            logger.info(f"Optimization completed. Pareto front size: {len(self.pareto_front.solutions)}")
            return self.pareto_front
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            raise
    
    async def _initialize_population(self):
        """Initialize random population"""
        try:
            self.population = []
            
            for _ in range(self.config.population_size):
                # Generate random individual
                individual = {
                    "genes": np.random.random(10),  # Example: 10-dimensional problem
                    "objectives": [0.0, 0.0],  # Example: 2 objectives
                    "rank": 0,
                    "crowding_distance": 0.0,
                    "dominated_count": 0,
                    "dominated_solutions": []
                }
                
                self.population.append(individual)
            
            logger.info(f"Initialized population of {len(self.population)} individuals")
            
        except Exception as e:
            logger.error(f"Population initialization failed: {e}")
            raise
    
    async def _evaluate_fitness(self):
        """Evaluate fitness of all individuals"""
        try:
            if self.config.parallel_evaluation:
                # Parallel evaluation
                tasks = []
                for individual in self.population:
                    task = self._evaluate_individual(individual)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            else:
                # Sequential evaluation
                for individual in self.population:
                    await self._evaluate_individual(individual)
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            raise
    
    async def _evaluate_individual(self, individual: Dict[str, Any]):
        """Evaluate fitness of single individual"""
        try:
            if self.config.custom_fitness_function:
                # Use custom fitness function
                objectives = await self.config.custom_fitness_function(individual["genes"])
                individual["objectives"] = objectives
            else:
                # Use default fitness function (example)
                individual["objectives"] = self._default_fitness_function(individual["genes"])
            
        except Exception as e:
            logger.error(f"Individual evaluation failed: {e}")
            individual["objectives"] = [0.0, 0.0]
    
    def _default_fitness_function(self, genes: np.ndarray) -> List[float]:
        """Default fitness function (example)"""
        # This is a simple example - in practice, you would implement your specific fitness function
        obj1 = np.sum(genes ** 2)  # Minimize sum of squares
        obj2 = np.sum((genes - 1) ** 2)  # Minimize distance from 1
        return [obj1, obj2]
    
    async def _update_pareto_front(self):
        """Update Pareto front using non-dominated sorting"""
        try:
            # Calculate domination relationships
            self._calculate_domination()
            
            # Non-dominated sorting
            fronts = self._non_dominated_sorting()
            
            # Calculate crowding distances
            self._calculate_crowding_distances(fronts)
            
            # Update Pareto front
            self.pareto_front = ParetoFront(
                solutions=[ind["genes"] for ind in self.population],
                objectives=[ind["objectives"] for ind in self.population],
                ranks=[ind["rank"] for ind in self.population],
                crowding_distances=[ind["crowding_distance"] for ind in self.population]
            )
            
        except Exception as e:
            logger.error(f"Pareto front update failed: {e}")
            raise
    
    def _calculate_domination(self):
        """Calculate domination relationships"""
        for i, individual1 in enumerate(self.population):
            individual1["dominated_count"] = 0
            individual1["dominated_solutions"] = []
            
            for j, individual2 in enumerate(self.population):
                if i != j:
                    if self._dominates(individual1["objectives"], individual2["objectives"]):
                        individual1["dominated_solutions"].append(j)
                    elif self._dominates(individual2["objectives"], individual1["objectives"]):
                        individual1["dominated_count"] += 1
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2"""
        # For minimization problems
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    def _non_dominated_sorting(self) -> List[List[int]]:
        """Perform non-dominated sorting"""
        fronts = []
        remaining = list(range(len(self.population)))
        
        while remaining:
            current_front = []
            next_remaining = []
            
            for i in remaining:
                if self.population[i]["dominated_count"] == 0:
                    current_front.append(i)
                    self.population[i]["rank"] = len(fronts)
                else:
                    next_remaining.append(i)
            
            if not current_front:
                break
            
            fronts.append(current_front)
            
            # Update domination counts
            for i in current_front:
                for j in self.population[i]["dominated_solutions"]:
                    self.population[j]["dominated_count"] -= 1
            
            remaining = next_remaining
        
        return fronts
    
    def _calculate_crowding_distances(self, fronts: List[List[int]]):
        """Calculate crowding distances for each front"""
        for front in fronts:
            if len(front) <= 2:
                # If front has 2 or fewer solutions, set crowding distance to infinity
                for i in front:
                    self.population[i]["crowding_distance"] = float('inf')
            else:
                # Calculate crowding distance for each objective
                for obj_idx in range(len(self.population[0]["objectives"])):
                    # Sort by objective value
                    front_sorted = sorted(front, key=lambda x: self.population[x]["objectives"][obj_idx])
                    
                    # Set boundary solutions to infinity
                    self.population[front_sorted[0]]["crowding_distance"] = float('inf')
                    self.population[front_sorted[-1]]["crowding_distance"] = float('inf')
                    
                    # Calculate crowding distance for intermediate solutions
                    obj_range = (self.population[front_sorted[-1]]["objectives"][obj_idx] - 
                               self.population[front_sorted[0]]["objectives"][obj_idx])
                    
                    if obj_range > 0:
                        for i in range(1, len(front_sorted) - 1):
                            idx = front_sorted[i]
                            distance = (self.population[front_sorted[i + 1]]["objectives"][obj_idx] - 
                                      self.population[front_sorted[i - 1]]["objectives"][obj_idx]) / obj_range
                            self.population[idx]["crowding_distance"] += distance
    
    async def _selection(self) -> List[Dict[str, Any]]:
        """Select parents for reproduction"""
        try:
            if self.config.algorithm == OptimizationAlgorithm.NSGA_II:
                return await self._nsga_ii_selection()
            elif self.config.algorithm == OptimizationAlgorithm.NSGA_III:
                return await self._nsga_iii_selection()
            elif self.config.algorithm == OptimizationAlgorithm.SPEA2:
                return await self._spea2_selection()
            else:
                raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
                
        except Exception as e:
            logger.error(f"Selection failed: {e}")
            raise
    
    async def _nsga_ii_selection(self) -> List[Dict[str, Any]]:
        """NSGA-II selection"""
        # Sort population by rank and crowding distance
        sorted_population = sorted(
            self.population,
            key=lambda x: (x["rank"], -x["crowding_distance"])
        )
        
        # Select best individuals
        selected = sorted_population[:self.config.population_size]
        
        return selected
    
    async def _nsga_iii_selection(self) -> List[Dict[str, Any]]:
        """NSGA-III selection"""
        # This is a simplified implementation
        # In practice, you would implement the full NSGA-III algorithm
        return await self._nsga_ii_selection()
    
    async def _spea2_selection(self) -> List[Dict[str, Any]]:
        """SPEA2 selection"""
        # This is a simplified implementation
        # In practice, you would implement the full SPEA2 algorithm
        return await self._nsga_ii_selection()
    
    async def _crossover(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform crossover to create offspring"""
        try:
            offspring = []
            
            # Pair parents
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1 = parents[i]
                    parent2 = parents[i + 1]
                    
                    # Perform crossover with probability
                    if random.random() < self.config.crossover_rate:
                        child1, child2 = self._uniform_crossover(parent1, parent2)
                        offspring.extend([child1, child2])
                    else:
                        # No crossover, copy parents
                        offspring.extend([parent1, child2])
            
            return offspring
            
        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            raise
    
    def _uniform_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover"""
        child1_genes = np.zeros_like(parent1["genes"])
        child2_genes = np.zeros_like(parent2["genes"])
        
        for i in range(len(parent1["genes"])):
            if random.random() < 0.5:
                child1_genes[i] = parent1["genes"][i]
                child2_genes[i] = parent2["genes"][i]
            else:
                child1_genes[i] = parent2["genes"][i]
                child2_genes[i] = parent1["genes"][i]
        
        child1 = {
            "genes": child1_genes,
            "objectives": [0.0, 0.0],
            "rank": 0,
            "crowding_distance": 0.0,
            "dominated_count": 0,
            "dominated_solutions": []
        }
        
        child2 = {
            "genes": child2_genes,
            "objectives": [0.0, 0.0],
            "rank": 0,
            "crowding_distance": 0.0,
            "dominated_count": 0,
            "dominated_solutions": []
        }
        
        return child1, child2
    
    async def _mutation(self, offspring: List[Dict[str, Any]]):
        """Apply mutation to offspring"""
        try:
            for individual in offspring:
                if random.random() < self.config.mutation_rate:
                    self._gaussian_mutation(individual)
            
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            raise
    
    def _gaussian_mutation(self, individual: Dict[str, Any]):
        """Gaussian mutation"""
        mutation_strength = 0.1  # Adjust as needed
        noise = np.random.normal(0, mutation_strength, individual["genes"].shape)
        individual["genes"] += noise
        
        # Ensure genes stay within bounds
        individual["genes"] = np.clip(individual["genes"], 0, 1)
    
    async def _replacement(self, offspring: List[Dict[str, Any]]):
        """Replace population with offspring"""
        try:
            # Combine parents and offspring
            combined = self.population + offspring
            
            # Sort by rank and crowding distance
            combined.sort(key=lambda x: (x["rank"], -x["crowding_distance"]))
            
            # Select best individuals for next generation
            self.population = combined[:self.config.population_size]
            
        except Exception as e:
            logger.error(f"Replacement failed: {e}")
            raise
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume metric"""
        if not self.pareto_front or not self.pareto_front.solutions:
            return 0.0
        
        # This is a simplified implementation
        # In practice, you would use a proper hypervolume calculation library
        objectives = np.array(self.pareto_front.objectives)
        
        # Calculate hypervolume (simplified)
        if len(objectives) == 0:
            return 0.0
        
        # For 2D case, calculate area
        if objectives.shape[1] == 2:
            # Sort by first objective
            sorted_obj = objectives[objectives[:, 0].argsort()]
            
            # Calculate area under the curve
            area = 0.0
            for i in range(len(sorted_obj) - 1):
                area += (sorted_obj[i + 1, 0] - sorted_obj[i, 0]) * sorted_obj[i, 1]
            
            return area
        
        return 0.0
    
    def _calculate_spread(self) -> float:
        """Calculate spread metric"""
        if not self.pareto_front or not self.pareto_front.solutions:
            return 0.0
        
        objectives = np.array(self.pareto_front.objectives)
        
        if len(objectives) == 0:
            return 0.0
        
        # Calculate spread (simplified)
        # This measures the distribution of solutions along the Pareto front
        distances = []
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                distance = np.linalg.norm(objectives[i] - objectives[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        # Check hypervolume improvement
        if len(self.hypervolume_history) >= 2:
            improvement = abs(self.hypervolume_history[-1] - self.hypervolume_history[-2])
            if improvement < self.config.convergence_threshold:
                return True
        
        # Check spread improvement
        if len(self.spread_history) >= 2:
            improvement = abs(self.spread_history[-1] - self.spread_history[-2])
            if improvement < self.config.convergence_threshold:
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "pareto_front_size": len(self.pareto_front.solutions) if self.pareto_front else 0,
            "hypervolume": self.hypervolume_history[-1] if self.hypervolume_history else 0.0,
            "spread": self.spread_history[-1] if self.spread_history else 0.0,
            "converged": self.converged
        }
