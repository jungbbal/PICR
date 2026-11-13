"""Checkpoint loading, open-loop rollout evaluation, and rendering.

Deliberately empty of re-exports. The old package re-exported every symbol here,
which made ``import ...eval`` pull matplotlib and the video writer into any process
that only wanted to load a checkpoint. Import the submodule you need:

    from cdelan.eval.rollout import load_checkpoint_model
    from cdelan.eval.episode import per_horizon_episode_errors
"""
