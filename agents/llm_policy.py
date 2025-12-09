"""
LLM-Based Agent Policy

Wraps LLM calls to make contribution decisions based on game state and memory.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from environment.public_goods_env import RoundObservation
from memory.base_memory import BaseMemory, MemoryState, RoundInfo
from agents.base_policy import AgentPolicy, AgentAction


@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMBackend(ABC):
    """Abstract interface for LLM backends."""
    
    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass


class MockLLMBackend(LLMBackend):
    """
    Mock LLM backend for testing without API calls.
    
    Implements simple heuristic-based responses that mimic LLM behavior.
    """
    
    def __init__(self, strategy: str = "cooperative", noise: float = 0.1) -> None:
        self.strategy = strategy
        self.noise = noise
        self._rng_seed = 42
        import random
        self._rng = random.Random(self._rng_seed)
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        # Parse key info from prompt
        round_match = re.search(r"Round (\d+)", prompt)
        budget_match = re.search(r"budget[:\s]+(\d+\.?\d*)", prompt, re.IGNORECASE)
        max_match = re.search(r"maximum[:\s]+(\d+)", prompt, re.IGNORECASE)
        
        current_round = int(round_match.group(1)) if round_match else 1
        budget = float(budget_match.group(1)) if budget_match else 20.0
        max_contrib = int(max_match.group(1)) if max_match else 10
        
        # Determine contribution based on strategy
        if self.strategy == "cooperative":
            base = int(max_contrib * 0.7)
        elif self.strategy == "defector":
            base = int(max_contrib * 0.2)
        elif self.strategy == "tit_for_tat":
            # Look for trust scores or previous contributions
            trust_match = re.search(r"Trust.*?(\d+\.\d+)", prompt)
            if trust_match:
                trust = float(trust_match.group(1))
                base = int(max_contrib * trust)
            else:
                base = int(max_contrib * 0.5)
        else:
            base = int(max_contrib * 0.5)
        
        # Add noise
        noise_amt = int(self._rng.gauss(0, self.noise * max_contrib))
        contribution = max(0, min(int(budget), max_contrib, base + noise_amt))
        
        # Generate mock response
        reasoning = self._generate_reasoning(contribution, current_round)
        
        response_json = {
            "contribution": contribution,
            "reasoning": reasoning,
        }
        
        if "summary" in prompt.lower():
            response_json["updated_summary"] = f"Round {current_round}: contributed {contribution}. Watching others' behavior."
        
        if "strategy note" in prompt.lower():
            response_json["strategy_note"] = "Maintain moderate cooperation while watching for defectors."
        
        content = f"```json\n{json.dumps(response_json, indent=2)}\n```"
        
        # Estimate tokens
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(content) // 4
        
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    
    def _generate_reasoning(self, contribution: int, round_num: int) -> str:
        if contribution >= 7:
            return f"Round {round_num}: Contributing high ({contribution}) to encourage cooperation and maximize group welfare."
        elif contribution >= 4:
            return f"Round {round_num}: Contributing moderately ({contribution}) to balance personal gain with group benefit."
        else:
            return f"Round {round_num}: Contributing conservatively ({contribution}) to protect my resources."


class OpenAIBackend(LLMBackend):
    """OpenAI API backend for LLM calls."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content or ""
        usage = response.usage
        
        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


class LLMPolicy(AgentPolicy):
    """
    LLM-based agent policy.
    
    Uses an LLM to make contribution decisions based on:
    - Game rules and current state
    - Memory representation (varies by condition)
    - Previous round outcomes
    """
    
    SYSTEM_PROMPT = """You are playing a repeated public goods game with other agents.

GAME RULES:
- Each round, you and {num_agents} other agents each choose how many coins to contribute to a shared pot.
- The total pot is multiplied by {multiplier}x and split equally among all agents.
- Your goal is to maximize your final coin total across all rounds.

STRATEGIC CONSIDERATIONS:
- If everyone contributes maximally, everyone benefits from the multiplier.
- If you contribute less while others contribute more, you gain more personally.
- If everyone defects, everyone loses the benefit of the multiplier.
- This creates a tension between individual and collective rationality.

RESPONSE FORMAT:
You must respond with valid JSON containing:
- "contribution": integer (your chosen contribution for this round)
- "reasoning": string (brief explanation of your choice)
{extra_fields}

Example:
```json
{{
  "contribution": 5,
  "reasoning": "Contributing moderately to maintain cooperation while protecting resources."
}}
```"""

    def __init__(
        self,
        backend: LLMBackend,
        memory: BaseMemory,
        num_agents: int = 3,
        multiplier: float = 1.8,
        max_retries: int = 3,
        default_contribution: int = 3,
    ) -> None:
        self.backend = backend
        self.memory = memory
        self.num_agents = num_agents
        self.multiplier = multiplier
        self.max_retries = max_retries
        self.default_contribution = default_contribution
        
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
    
    @property
    def name(self) -> str:
        return f"LLM({self.memory.name})"
    
    @property
    def total_tokens(self) -> tuple[int, int]:
        """Return (prompt_tokens, completion_tokens) used so far."""
        return self._total_prompt_tokens, self._total_completion_tokens
    
    def reset(self) -> None:
        """Reset token counters for new episode."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
    
    def act(
        self,
        observation: RoundObservation,
        memory_state: MemoryState,
        max_contribution: int,
    ) -> AgentAction:
        # Build prompt
        prompt = self._build_prompt(observation, memory_state, max_contribution)
        
        # Get extra fields for system prompt
        extra_fields = self._get_extra_response_fields()
        system_prompt = self.SYSTEM_PROMPT.format(
            num_agents=self.num_agents,
            multiplier=self.multiplier,
            extra_fields=extra_fields,
        )
        
        # Try to get valid response
        for attempt in range(self.max_retries):
            try:
                response = self.backend.complete(prompt, system_prompt)
                self._total_prompt_tokens += response.prompt_tokens
                self._total_completion_tokens += response.completion_tokens
                
                action = self._parse_response(response.content, observation, max_contribution)
                action.prompt_tokens = response.prompt_tokens
                action.completion_tokens = response.completion_tokens
                action.raw_response = response.content
                
                return action
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    # Final fallback
                    return AgentAction(
                        contribution=min(self.default_contribution, max_contribution, int(observation.budget)),
                        reasoning=f"Fallback after {self.max_retries} failed attempts: {e}",
                    )
        
        # Should not reach here
        return AgentAction(
            contribution=self.default_contribution,
            reasoning="Unexpected fallback",
        )
    
    def _build_prompt(
        self,
        observation: RoundObservation,
        memory_state: MemoryState,
        max_contribution: int,
    ) -> str:
        lines = [
            f"=== CURRENT STATE (Round {observation.round_index + 1}) ===",
            f"Your remaining budget: {observation.budget:.1f} coins",
            f"Your cumulative payoff: {observation.cumulative_payoff:+.1f}",
            f"Maximum contribution this round: {min(max_contribution, int(observation.budget))}",
            "",
        ]
        
        # Add last round info if available
        if observation.last_round_contributions is not None:
            lines.append("=== LAST ROUND RESULTS ===")
            for i, c in enumerate(observation.last_round_contributions):
                if i == observation.agent_id:
                    lines.append(f"  You contributed: {c}")
                else:
                    lines.append(f"  Agent {i} contributed: {c}")
            if observation.last_round_pot is not None:
                lines.append(f"  Total pot (after {self.multiplier}x multiplier): {observation.last_round_pot:.1f}")
            if observation.last_round_payoffs is not None:
                lines.append(f"  Your payoff: {observation.last_round_payoffs[observation.agent_id]:+.1f}")
            lines.append("")
        
        # Add memory
        memory_text = self.memory.render_for_prompt(memory_state)
        lines.append(memory_text)
        lines.append("")
        
        # Add decision prompt
        lines.append("=== YOUR DECISION ===")
        lines.append(f"Choose your contribution (0 to {min(max_contribution, int(observation.budget))}).")
        lines.append("Respond with JSON as specified.")
        
        return "\n".join(lines)
    
    def _get_extra_response_fields(self) -> str:
        """Get additional response fields based on memory type."""
        from memory.summary_memory import SummaryMemory
        from memory.hybrid_memory import HybridMemory
        
        extra = ""
        if isinstance(self.memory, SummaryMemory):
            extra = '- "updated_summary": string (updated memory summary, max 50 words)\n'
        elif isinstance(self.memory, HybridMemory):
            extra = '- "strategy_note": string (updated strategy note, max 20 words)\n'
        
        return extra
    
    def _parse_response(
        self,
        content: str,
        observation: RoundObservation,
        max_contribution: int,
    ) -> AgentAction:
        # Extract JSON from response
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        data = json.loads(json_str)
        
        contribution = int(data["contribution"])
        contribution = max(0, min(contribution, max_contribution, int(observation.budget)))
        
        reasoning = data.get("reasoning", "")
        
        return AgentAction(
            contribution=contribution,
            reasoning=reasoning,
        )
    
    def extract_memory_update(self, response_content: str) -> Optional[str]:
        """Extract memory update content from LLM response."""
        try:
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("updated_summary") or data.get("strategy_note")
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

