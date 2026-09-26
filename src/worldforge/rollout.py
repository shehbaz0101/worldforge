"""Roll a dynamics environment into one trajectory.

``zero`` applies a zero dose. ``random`` draws each dose component uniformly
inside the environment's action bounds from ``random.Random(seed)``. That
generator is separate from the environment's initial-state generator, so the
action sequence does not depend on how ``reset`` consumes its seed. Both
policies are deterministic given the seed. Nothing here uses the network.
"""

from __future__ import annotations

import random
from typing import Literal

from worldforge.envs.protocol import DynamicsEnv
from worldforge.schemas import Action, Observation, Trajectory, Transition

Policy = Literal["zero", "random"]


def rollout(
    env: DynamicsEnv,
    *,
    n_steps: int,
    seed: int,
    policy: Policy = "zero",
) -> Trajectory:
    """Step ``env`` up to ``n_steps`` times and return the episode.

    The environment's own horizon can stop the episode earlier. In that case
    ``Trajectory.horizon`` is still the requested ``n_steps``, and the last
    stored transition carries ``truncated`` or ``terminated``.
    """

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be an int >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if policy not in ("zero", "random"):
        raise ValueError("policy must be 'zero' or 'random'")

    observation = env.reset(seed)
    if env.seed != seed:
        raise RuntimeError("env returned a different seed than the one requested")
    rng = random.Random(seed)
    transitions: list[Transition] = []
    for _ in range(n_steps):
        action = _policy_action(env, policy, rng)
        t_before = env.t
        next_observation, reward, terminated, truncated, _info = env.step(action)
        transitions.append(
            Transition(
                observation=Observation(values=[float(value) for value in observation], t=t_before),
                action=Action(values=[float(value) for value in action]),
                reward=float(reward),
                next_observation=Observation(
                    values=[float(value) for value in next_observation],
                    t=env.t,
                ),
                terminated=terminated,
                truncated=truncated,
            )
        )
        observation = next_observation
        if terminated or truncated:
            break
    return Trajectory(
        env_id=env.env_id,
        seed=seed,
        horizon=n_steps,
        transitions=transitions,
    )


def format_summary(trajectory: Trajectory, *, policy: str) -> str:
    """Plain-text episode summary for the Day 1 CLI.

    State components are ``x0``, ``x1``, ... so the helper stays valid when
    another environment registers a different observation size. The default
    environment's prey and predator are ``x0`` and ``x1``.
    """

    if trajectory.transitions:
        first = trajectory.transitions[0].observation
        last = trajectory.transitions[-1]
        initial = _format_observation(first)
        final = _format_observation(last.next_observation)
        total_reward = sum(item.reward for item in trajectory.transitions)
        # ``-0.0`` prints with a minus. An exact zero return should read as 0.
        if total_reward == 0.0:
            total_reward = 0.0
        terminated = last.terminated
        truncated = last.truncated
    else:
        initial = "n/a"
        final = "n/a"
        total_reward = 0.0
        terminated = False
        truncated = False
    lines = [
        f"env_id: {trajectory.env_id}",
        f"seed: {trajectory.seed}",
        f"policy: {policy}",
        f"horizon: {trajectory.horizon}",
        f"steps: {len(trajectory.transitions)}",
        f"return: {total_reward:.6f}",
        f"initial: {initial}",
        f"final: {final}",
        f"terminated: {str(terminated).lower()}",
        f"truncated: {str(truncated).lower()}",
    ]
    return "\n".join(lines)


def _policy_action(
    env: DynamicsEnv,
    policy: Policy,
    rng: random.Random,
) -> tuple[float, ...]:
    if policy == "zero":
        return tuple(0.0 for _ in range(env.action_dim))
    if len(env.action_low) != env.action_dim or len(env.action_high) != env.action_dim:
        raise ValueError("action bounds must match action_dim")
    return tuple(
        rng.uniform(low, high)
        for low, high in zip(env.action_low, env.action_high, strict=True)
    )


def _format_observation(observation: Observation) -> str:
    parts = " ".join(
        f"x{index}={value:.6f}" for index, value in enumerate(observation.values)
    )
    return f"t={observation.t:.2f} {parts}"
