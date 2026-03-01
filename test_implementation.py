"""
Quick Test - Verify RL Implementation Works
Run this first to check everything is installed correctly
"""

import sys
sys.path.append('/mnt/user-data/uploads')

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        from scipy.integrate import odeint
        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False
    
    try:
        import pygame
        print("✓ pygame")
    except ImportError as e:
        print(f"✗ pygame: {e}")
        return False
    
    try:
        from pettingzoo import ParallelEnv
        print("✓ pettingzoo")
    except ImportError as e:
        print(f"✗ pettingzoo: {e}")
        return False
    
    try:
        from gymnasium.spaces import Discrete, Box
        print("✓ gymnasium")
    except ImportError as e:
        print(f"✗ gymnasium: {e}")
        return False
    
    return True


def test_environment():
    """Test environment creation and reset"""
    print("\nTesting environment...")
    
    try:
        from rl_environment import PredatorPreyEnv
        
        env = PredatorPreyEnv(render_mode=None, num_prey=2, num_predator=2)
        print("✓ Environment created")
        
        obs, info = env.reset()
        print(f"✓ Environment reset: {len(obs)} agents")
        
        # Check observation shapes
        for agent, observation in obs.items():
            expected_shape = 36 if 'prey' in agent else 15
            if observation.shape[0] == expected_shape:
                print(f"✓ {agent} observation shape correct: {observation.shape}")
            else:
                print(f"✗ {agent} observation shape wrong: {observation.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Environment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_q_learning():
    """Test Q-learning agent"""
    print("\nTesting Q-learning agent...")
    
    try:
        from q_learning import QLearningAgent
        import numpy as np
        
        agent = QLearningAgent(agent_type='prey')
        print("✓ Q-learning agent created")
        
        # Test action selection
        obs = np.random.rand(36)
        action = agent.get_action(obs, training=True)
        print(f"✓ Action selected: {action}")
        
        # Test Q-value update
        next_obs = np.random.rand(36)
        agent.update(obs, action, 1.0, next_obs, False)
        print("✓ Q-value updated")
        
        return True
        
    except Exception as e:
        print(f"✗ Q-learning error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step():
    """Test environment step"""
    print("\nTesting environment step...")
    
    try:
        from rl_environment import PredatorPreyEnv
        import numpy as np
        
        env = PredatorPreyEnv(render_mode=None, num_prey=2, num_predator=2)
        obs, _ = env.reset()
        
        # Random actions
        actions = {}
        for agent in obs:
            if 'prey' in agent:
                actions[agent] = np.random.randint(0, 7)
            else:
                actions[agent] = np.random.randint(0, 9)
        
        new_obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"✓ Step executed: {len(new_obs)} agents remaining")
        print(f"✓ Rewards: {list(rewards.values())[:3]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Step error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_training():
    """Test a very short training run"""
    print("\nTesting short training run (10 episodes)...")
    
    try:
        from rl_environment import PredatorPreyEnv
        from q_learning import MultiAgentQLearning
        
        env = PredatorPreyEnv(render_mode=None, num_prey=2, num_predator=2)
        
        prey_agents = ["prey_0", "prey_1"]
        predator_agents = ["predator_0", "predator_1"]
        
        ql_manager = MultiAgentQLearning(
            prey_agents=prey_agents,
            predator_agents=predator_agents,
            learning_rate=0.1,
            discount_factor=0.9
        )
        
        print("✓ QL manager created")
        
        for episode in range(10):
            obs, _ = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                actions = ql_manager.get_actions(obs, training=True)
                new_obs, rewards, terms, truncs, _ = env.step(actions)
                
                # Update Q-values
                for agent in obs:
                    if agent in actions and agent in rewards:
                        next_o = new_obs.get(agent, obs[agent])
                        ql_manager.update(
                            agent, obs[agent], actions[agent],
                            rewards[agent], next_o, terms.get(agent, False)
                        )
                
                obs = new_obs
                steps += 1
                
                if len(obs) == 0:
                    done = True
            
            ql_manager.decay_epsilon()
        
        stats = ql_manager.get_statistics()
        print(f"✓ Training complete: epsilon={stats['epsilon']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("EvolveRL Implementation Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        print("\n❌ Import test failed!")
        print("Run: pip install numpy matplotlib scipy pygame pettingzoo gymnasium tqdm --break-system-packages")
        all_passed = False
    
    if all_passed and not test_environment():
        print("\n❌ Environment test failed!")
        all_passed = False
    
    if all_passed and not test_q_learning():
        print("\n❌ Q-learning test failed!")
        all_passed = False
    
    if all_passed and not test_step():
        print("\n❌ Step test failed!")
        all_passed = False
    
    if all_passed and not test_short_training():
        print("\n❌ Training test failed!")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python train_rl.py          # Train the agents")
        print("  python demo_agents.py       # Demo trained agents")
        print("  python demo_agents.py compare  # Compare performance")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the errors above before proceeding.")
    print("=" * 60)
